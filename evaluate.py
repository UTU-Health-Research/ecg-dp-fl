#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import torch
import numpy as np
import json

from src.modeling.models import seresnet18
from src.modeling.train_utils import Predict, ModelBase
from src.modeling.metrics import cal_multilabel_metrics
from src.dataloader.dataset import dataloader
from src.dataloader.features import test_set, load_split_features, metadata_hash
from src.dataloader.silo import dataset_splits, get_silos, get_experiment_seed

#SUBSET_NAME = "small"
SUBSET_NAME = "full"

BATCH_SIZE = 256
IN_CHANNELS = 12

ECG_DIR = "data/processed_data"
CSV_DIR = "data/split_csvs"

# relative to model path (hardcoded)
CHECKPOINT_PATH = "checkpoints"
FEATURES_PATH = "ext_features"
LOG_DIR = "logs"
METADATA = "metadata.json"
SPLITS_FILE = "splits.json"
FINETUNE_INDEX = "finetune.json"
FINETUNED_PATH = "finetune"
PRED_PATH = "predictions"

def run_predictor(model, test_ds, labels, weights, device="cuda"):
    p = Predict(model, test_ds, labels, device)
    p.set_weights(weights)
    hist = p.predict_proba()

    _, _, test_macro_auroc, test_micro_auroc = cal_multilabel_metrics(hist["labels_all"], hist["logits_prob_all"], labels, 0.5)
    print("eval AUC: {:.2f}".format(test_macro_auroc))
    hist['test_micro_auroc'] = test_micro_auroc
    hist['test_macro_auroc'] = test_macro_auroc
    return hist

# predict using the full model
def model_split_preds_full(splits, split_num, model, model_path, labels, device="cuda"):
    test_ds = dataloader(splits["test"][split_num], train=False, device=device, batch_size=BATCH_SIZE)
    with np.load(model_path) as npz:
        pretrained_weights = [npz[arr_name] for arr_name in npz.files]
    n_classes = pretrained_weights[-1].shape[0]
    if n_classes != len(labels):
        print("{} has {} channels, dataset is {} channels".format(
            model_path, n_classes, len(labels)))
        return {}
    return run_predictor(model, test_ds, labels, pretrained_weights, device)


# predict on extracted features using a 1-layer classifier
def model_split_preds(split, model_path, device="cuda", strategy="classif", model_arch="v4"):
    out_channels, test_ds = test_set(split, strategy, device, batch_size=BATCH_SIZE)

    with np.load(model_path) as npz:
        classif_ft_weights = [npz[arr_name] for arr_name in npz.files]
    if classif_ft_weights[-1].shape[0] != out_channels:
        print("{} has {} channels, dataset is {} channels".format(
            model_path, classif_ft_weights[-1].shape[0], out_channels))
        return {}
    model = seresnet18.finetuning_model(strategy, out_channels, model_arch)

    labels = [str(dummy) for dummy in range(out_channels)] # actual labels are not used anywhere
    return run_predictor(model, test_ds, labels, classif_ft_weights, device)

def save_preds(silo_name, out_path, split_num, train_set, hist):
    path_pref = os.path.join(out_path, silo_name)
    os.makedirs(path_pref, exist_ok=True)
    out_fn = f"preds_{train_set}_split{split_num}.npz"
    np.savez_compressed(os.path.join(path_pref, out_fn), y_hat=hist["logits_all"].cpu().numpy())
    return out_fn

def save_truth(silo_name, out_path, split_num, y_true):
    path_pref = os.path.join(out_path, silo_name)
    os.makedirs(path_pref, exist_ok=True)
    out_fn = f"y_true_split{split_num}.npz"
    np.savez_compressed(os.path.join(path_pref, out_fn), y_true=y_true)
    return out_fn

def get_meta_file(fn):
    metadata = {}
    if os.path.isfile(fn):
        with open(fn) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    return metadata

def run_finetuned_mode(args, model_dir, out_path, device="cuda"):
    ft_path = os.path.join(model_dir, FEATURES_PATH)
    clf_path = os.path.join(model_dir, FINETUNED_PATH)

    split_meta_file = os.path.join(model_dir, LOG_DIR, SPLITS_FILE)
    splits_meta = get_meta_file(split_meta_file)
    if not splits_meta:
        print("Cannot evaluate finetuned models, feature vectors not found")
        return None

    ft_meta_file = os.path.join(model_dir, LOG_DIR, FINETUNE_INDEX)
    ft_meta = get_meta_file(ft_meta_file)
    if not ft_meta:
        print("Finetuned models not found")
        return None

    summary_data = {"pretrained": {}}
    filenames = {"pretrained": {}, "y_true": {}}
    for strategy, s_splits in splits_meta[args.silo]["splits"].items():
        k = len(s_splits)
        summary_data[strategy] = {}
        filenames[strategy] = {}

        for split_num in range(k):
            split, test_y = load_split_features(ft_path, splits_meta[args.silo], split_num, strategy)

            ft_model_key = str(split_num)
            model_fn = ft_meta[args.silo]["splits"][args.train_set][strategy][ft_model_key]
            print(model_fn, args.silo, split_num, ft_model_key)
            hist = model_split_preds(split, os.path.join(clf_path, args.silo, model_fn),
                    device=device, strategy=strategy, model_arch=ft_meta[args.silo]["model_arch"])
            filename = save_preds(args.silo, out_path, split_num, f"{strategy}_{args.train_set}", hist)
            summary_data[strategy][str(split_num)] = hist["test_macro_auroc"]
            filenames[strategy][str(split_num)] = filename

            if strategy == "classif":
                filenames["y_true"][str(split_num)] = save_truth(args.silo, out_path, split_num, test_y)

                model_fn = ft_meta[args.silo]["splits"][args.train_set].get("default")
                if model_fn is None:
                    continue # can happen with stacked ensembles,
                             # which do not have their own classifier

                print(model_fn, args.silo, split_num, "default")
                hist = model_split_preds(split, os.path.join(clf_path, args.silo, model_fn),
                        device=device, strategy="classif", model_arch=ft_meta[args.silo]["model_arch"])
                if not hist:
                    continue
                filename = save_preds(args.silo, out_path, split_num, "pretrained", hist)
                summary_data["pretrained"][str(split_num)] = hist["test_macro_auroc"]
                filenames["pretrained"][str(split_num)] = filename

    return summary_data, filenames

def run_full_mode(args, model_dir, model_metadata, out_path, device="cuda"):
    silos, _ = get_silos(os.path.join(args.storage, CSV_DIR),
                          os.path.join(args.storage, ECG_DIR),
                          subset=args.subset_name)
    cinc_labels = silos[args.silo].columns.tolist()[4:]

    if type(model_metadata["model_file"]) == dict:
        model_file = model_metadata["model_file"].get(args.silo)
        if model_file is None:
            print("no personalized model for silo {}".format(args.silo))
            sys.exit(1)
    else:
        model_file = model_metadata["model_file"]

    model_name = metadata_hash(model_metadata)
    cp_path = os.path.join(model_dir, CHECKPOINT_PATH, model_file)
    random_seed = get_experiment_seed(args.silo, model_metadata["experiment_id"])
    model = seresnet18.resnet18(random_seed, model_metadata["model_arch"],
            in_channel=IN_CHANNELS, out_channel=len(cinc_labels))

    if args.num_splits > 1:
        spl = dataset_splits(args.silo, silos, k=args.num_splits, random_state=random_seed)
    else:
        spl = {"test": [silos[args.silo]]}

    summary_data = {"pretrained": {}}
    filenames = {"pretrained": {}, "y_true": {}}
    for split_num in range(args.num_splits):
        print(model_name, args.silo, split_num)
        hist = model_split_preds_full(spl, split_num, model, cp_path, cinc_labels, device=device)
        if not hist:
            break
        filename = save_preds(args.silo, out_path, split_num, "pretrained", hist)
        filenames["pretrained"][str(split_num)] = filename
        summary_data["pretrained"][str(split_num)] = hist["test_macro_auroc"]
        filenames["y_true"][str(split_num)] = save_truth(args.silo, out_path, split_num,
                hist["labels_all"].detach().cpu().numpy())
    return summary_data, filenames

TEST_SILO = "G12EC"
#TEST_SILO = "Code15"
#TEST_SILO = "Hefei"
K = 5

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", help="data and model files location",
            default=".", type=str)
    parser.add_argument("--model-dir", help="model subdirectory (relative to storage)",
            default="", type=str)
    parser.add_argument("--mode", help="evaluation mode (full|finetuned)",
            default="finetuned", type=str)
    parser.add_argument("--silo", help="hospital to evaluate",
            default=TEST_SILO, type=str)
    parser.add_argument("--train-set", help="train set name (train/train_1K), for finetuned models",
            default="train", type=str)
    parser.add_argument("--num-splits", help="number of splits to generate, ignored for finetuned models",
            default=K, type=int)
    parser.add_argument("--subset-name", help="subset to evaluate on (full/small), when splits are generated on the fly",
            default=SUBSET_NAME, type=str)
    args = parser.parse_args()

    if not args.model_dir:
        print("--model-dir parameter is mandatory")
        sys.exit(1)

    print("using", args.storage, "for input")
    model_dir = os.path.join(args.storage, args.model_dir)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(os.path.join(model_dir, LOG_DIR, METADATA)) as f:
        model_metadata = json.load(f)
    out_path = os.path.join(model_dir, PRED_PATH)

    evals_file = os.path.join(model_dir, LOG_DIR, "evaluation.json")
    evals_summary = get_meta_file(evals_file)

    if args.mode == "full":
        new_evals, new_fn = run_full_mode(args, model_dir, model_metadata, out_path, device)
    else:
        new_evals = evals_summary.get(args.silo, {}).get("splits", {})
        new_fn = evals_summary.get(args.silo, {}).get("predictions", {})
        new_splits, pred_fn = run_finetuned_mode(args, model_dir, out_path, device)
        new_evals[args.train_set] = new_splits
        new_fn[args.train_set] = pred_fn

    evals_summary.update({args.silo: {
                "splits": new_evals,
                "predictions": new_fn,
                "silo_name": args.silo,
                "mode": args.mode,
                "experiment_id": model_metadata["experiment_id"]
            }
        })
    with open(evals_file, "w") as f:
        json.dump(evals_summary, f, indent=2)
        print("evaluation summary saved to", evals_file)

