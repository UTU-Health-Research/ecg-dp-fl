#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import torch
import numpy as np
import json

from src.modeling.models import seresnet18
from src.modeling.train_utils import Extract, ModelBase
from src.dataloader.silo import dataset_splits, get_silos, get_experiment_seed, single_split
from src.dataloader.dataset import dataloader
from src.dataloader.features import metadata_hash, save_model_split

#SUBSET_NAME = "small"
SUBSET_NAME = "full"

BATCH_SIZE=256
IN_CHANNELS=12
RANDOM_SEED=42   # for old model code compatibility, does not affect results

ECG_DIR = "data/processed_data"
CSV_DIR = "data/split_csvs"

# relative to model path (hardcoded)
FEATURES_PATH = "ext_features"
CHECKPOINT_PATH = "checkpoints"
LOG_DIR = "logs"
METADATA = "metadata.json"
SPLITS_FILE = "splits.json"

# extract one model and one split
def extract_model_split(m, splits, split_num, labels, device="cuda"):
    def g2c(t):
        return t.detach().cpu().numpy()
    res = {}
    for k in splits:
        dl = dataloader(splits[k][split_num], train=False, batch_size=BATCH_SIZE)
        p = Extract(m, dl, labels, device)
        hist = p.predict_proba()
        res[k] = g2c(hist["logits_all"]), g2c(hist["ag_all"]), g2c(hist["labels_all"])
    return res

# extract all for one model
def extract_model_and_ds(model_name, ex_class, model_arch, model_path, silo_name, out_path,
        labels, spl, device="cuda", k=5, seed=RANDOM_SEED):
    with np.load(model_path) as npz:
        pt_weights = [npz[arr_name] for arr_name in npz.files]

    seresnet18.set_seed(seed)
    ex = ex_class(seresnet18.BasicBlock,
                    seresnet18.ARCH_PLANES[model_arch],
                    [2, 2, 2, 2],
                    in_channel=IN_CHANNELS, out_channel=len(labels))
    m = ModelBase(ex, labels, device)
    m.set_weights(pt_weights)

    split_data = {}
    for split_num in range(k):
        print(model_name, silo_name, split_num)
        feats = extract_model_split(m.model, spl, split_num, labels, device)
        filenames = save_model_split(model_name, silo_name, out_path, split_num, feats)
        split_data[str(split_num)] = filenames
    return split_data

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
    parser.add_argument("--num-splits", help="number of splits to generate",
            default=K, type=int)
    parser.add_argument("--silo", help="hospital to extract features for",
            default=TEST_SILO, type=str)
    parser.add_argument("--extractor", help="feature extractor (classif/res2x)",
            default="classif", type=str)
    parser.add_argument("--subset-name", help="subset to extract features for (full/small/stratX)",
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

    silos, cinc_labels = get_silos(os.path.join(args.storage, CSV_DIR),
                          os.path.join(args.storage, ECG_DIR),
                          subset=args.subset_name)

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

    in_path = os.path.join(model_dir, CHECKPOINT_PATH, model_file)
    out_path = os.path.join(model_dir, FEATURES_PATH)
    split_meta_file = os.path.join(model_dir, LOG_DIR, SPLITS_FILE)
    splits_meta = {}
    if os.path.isfile(split_meta_file):
        with open(split_meta_file) as f:
            splits_meta = json.load(f)

    random_seed = get_experiment_seed(args.silo, model_metadata["experiment_id"])
    if args.subset_name.startswith("strat"):
        print("stratified subset selected, generating features for the pre-existing split")
        test_silos, _ = get_silos(os.path.join(args.storage, CSV_DIR),
                              os.path.join(args.storage, ECG_DIR),
                              subset=args.subset_name + "_test")
        spl = single_split(silos[args.silo], test_silos[args.silo], random_state=random_seed)
        num_splits = 1
    else:
        spl = dataset_splits(args.silo, silos, k=args.num_splits, random_state=random_seed)
        num_splits = args.num_splits

    if args.extractor == "classif":
        ex_class = seresnet18.ExtractorAgeECG8
        model_name += "_classif"
    elif args.extractor == "res2x":
        ex_class = seresnet18.ExtractorECG6
        model_name += "_res2x"
    else:
        print("invalid extractor {}".format(args.extractor))
        sys.exit(1)

    new_split_meta = extract_model_and_ds(model_name, ex_class, model_metadata["model_arch"], in_path, args.silo, out_path,
            cinc_labels, spl, device, k=num_splits, seed=random_seed)
    curr_split_meta = splits_meta.get(args.silo, {}).get("splits", {})
    curr_split_meta.update({args.extractor: new_split_meta})
    splits_meta.update({args.silo: {
                "splits": curr_split_meta,
                "silo_name": args.silo,
                "subset_name": args.subset_name,
                "model_arch": model_metadata["model_arch"],  # duplicate from metadata for easier loading
                "experiment_id": model_metadata["experiment_id"]
            }
        })
    with open(split_meta_file, "w") as f:
        json.dump(splits_meta, f, indent=2)
        print("split metadata written to", split_meta_file)

