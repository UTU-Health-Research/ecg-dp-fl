#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import torch
import numpy as np
import json

from src.modeling.models.seresnet18 import set_seed, finetuning_model
from src.modeling.train_utils import TrainAndSelect, adamopt, sgdopt
from src.dataloader.features import metadata_hash, train_and_val_sets, load_split_features
from src.dataloader.silo import get_experiment_seed

BATCH_SIZE = 256
LEARNING_RATE = 0.003
RANDOM_SEED=42   # normally experiment based id is used instead

# relative to model path (hardcoded)
CHECKPOINT_PATH = "checkpoints"
FEATURES_PATH = "ext_features"
LOG_DIR = "logs"
METADATA = "metadata.json"
SPLITS_FILE = "splits.json"
FINETUNE_INDEX = "finetune.json"
FINETUNED_PATH = "finetune"

def save_default_classifier(model_name, pretrained_weights, silo_name, out_path):
    pretrained_classifier = pretrained_weights[-2:]
  
    path_pref = os.path.join(out_path, silo_name)
    os.makedirs(path_pref, exist_ok=True)
    out_fn = f"{model_name}_default_classifier.npz"
    np.savez(os.path.join(path_pref, out_fn), *pretrained_classifier)
    return {"default" : out_fn}
 
# finetune for one split
def finetune_model_split(split, split_num, optfunc, epochs, pt_weights,
        device="cuda", train_set="train",
        random_seed=RANDOM_SEED, strategy="classif", model_arch="v4"):
    out_channels, train_ds, val_ds = train_and_val_sets(split, train_set, strategy, device)

    set_seed(random_seed)
    model = finetuning_model(strategy, out_channels, model_arch)
    labels = [str(dummy) for dummy in range(out_channels)] # actual labels are not used anywhere
    t = TrainAndSelect(model, optfunc, train_ds, val_ds, labels, device, ".", name=f"ft_classifier_{strategy}_{split_num}",
            patience=5)
    t.update_weights(pt_weights)
    t.cp_freq = None 
    hist = t.train(epochs, log=False)
    print("epochs: {} final val loss: {:.3f} val AUC: {:.2f}".format(len(hist["val_loss"]),
        hist["val_loss"][-1], hist["val_macro_auroc"][-1]))
    return t.best_model

def save_split_classifier(model_name, silo_name, out_path, split_num, train_set, weights):
    path_pref = os.path.join(out_path, silo_name)
    os.makedirs(path_pref, exist_ok=True)
    out_fn = f"{model_name}_ft_classifier_{train_set}_split{split_num}.npz"
    np.savez(os.path.join(path_pref, out_fn), *weights)
    return out_fn

# finetune all splits for one model
def ft_split_classifiers(model_name, ft_path, silo_name, out_path,
        spl, optfunc, epochs, pt_weights, device="cuda", train_set="train", random_seed=RANDOM_SEED,
        model_arch="v4"):
    split_data = {}
    for strategy, s_splits in spl[silo_name]["splits"].items():
        k = len(s_splits)
        m_strat_name = model_name + "_" + strategy
        split_data[strategy] = {}
        for split_num in range(k):
            print(m_strat_name, silo_name, split_num)
            split, test_y = load_split_features(ft_path, spl[silo_name], split_num, strategy)
            weights = finetune_model_split(split, split_num, optfunc, epochs, pt_weights,
                    device=device, train_set=train_set,
                    random_seed=random_seed, strategy=strategy, model_arch=model_arch)
            filename = save_split_classifier(m_strat_name, silo_name, out_path, split_num, train_set, weights)
            split_data[strategy][str(split_num)] = filename
    return split_data

TEST_SILO = "G12EC"
#TEST_SILO = "Code15"
#TEST_SILO = "Hefei"

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", help="data and model files location",
            default=".", type=str)
    parser.add_argument("--model-dir", help="model subdirectory (relative to storage)",
            default="", type=str)
    parser.add_argument("--silo", help="hospital to finetune for",
            default=TEST_SILO, type=str)
    parser.add_argument("--train-set", help="train set name (train/train_1K)",
            default="train", type=str)
    parser.add_argument("--local-epoch", help="number of epochs to finetune",
            default=100, type=int)
    parser.add_argument("--optfunc", help="optimizer (adam|sgd)",
            default="adam", type=str)
    parser.add_argument("--lr", help="learning rate",
            default=LEARNING_RATE, type=float)
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
    model_name = metadata_hash(model_metadata)

    if type(model_metadata["model_file"]) == dict:
        model_file = model_metadata["model_file"].get(args.silo)
        if model_file is None:
            print("no personalized model for silo {}".format(args.silo))
            sys.exit(1)
    else:
        model_file = model_metadata["model_file"]

    cp_path = os.path.join(model_dir, CHECKPOINT_PATH, model_file)
    in_path = os.path.join(model_dir, FEATURES_PATH)
    out_path = os.path.join(model_dir, FINETUNED_PATH)

    split_meta_file = os.path.join(model_dir, LOG_DIR, SPLITS_FILE)
    splits_meta = {}
    if os.path.isfile(split_meta_file):
        with open(split_meta_file) as f:
            splits_meta = json.load(f)
    else:
        print(f"Run this first: feature_extract.py --model-dir={model_dir}")
        sys.exit(2)

    ft_meta_file = os.path.join(model_dir, LOG_DIR, FINETUNE_INDEX)
    ft_meta = {}
    curr_ft = {}
    if os.path.isfile(ft_meta_file):
        with open(ft_meta_file) as f:
            ft_meta = json.load(f)
            curr_ft = ft_meta.get(args.silo, {}).get("splits", {})

    if args.optfunc == "sgd":
        optfunc = lambda x: sgdopt(x, lr=args.lr)
        print("using SGD optimizer")
    else:
        optfunc = lambda x: adamopt(x, lr=args.lr)
        print("using Adam optimizer")

    if os.path.isfile(cp_path):
        with np.load(cp_path) as npz:
            pretrained_weights = [npz[arr_name] for arr_name in npz.files]
    else:
        print("checkpoint path {} not found".format(cp_path))
        sys.exit(1)

    new_ft_meta = save_default_classifier(model_name, pretrained_weights, args.silo, out_path)
    random_seed = get_experiment_seed(args.silo, model_metadata["experiment_id"])
    new_ft_meta.update(ft_split_classifiers(model_name, in_path, args.silo, out_path,
            splits_meta, optfunc, args.local_epoch, pretrained_weights, device,
            args.train_set, random_seed, model_metadata["model_arch"]))
    curr_ft[args.train_set] = new_ft_meta

    ft_meta.update({args.silo: {
                "splits": curr_ft,
                "silo_name": args.silo,
                "model_arch": model_metadata["model_arch"],  # duplicate from metadata for easier loading
                "experiment_id": model_metadata["experiment_id"]
            }
        })
    with open(ft_meta_file, "w") as f:
        json.dump(ft_meta, f, indent=2)
        print("finetuning metadata written to", ft_meta_file)

