#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import numpy as np
import json

from src.dataloader.features import metadata_hash, load_split_features, save_model_split

LOG_DIR = "logs"
FEATURES_PATH = "ext_features"
METADATA = "metadata.json"
SPLITS_FILE = "splits.json"

def get_meta_file(fn):
    metadata = {}
    if os.path.isfile(fn):
        with open(fn) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    return metadata

def load_features(args, model_dirs):
    old_metadata = None
    features = {}
    for model_dir in model_dirs:
        with open(os.path.join(args.storage, model_dir, LOG_DIR, SPLITS_FILE)) as f:
            spl = json.load(f)

        with open(os.path.join(args.storage, model_dir, LOG_DIR, METADATA)) as f:
            model_metadata = json.load(f)
        if old_metadata is None:
            old_metadata = model_metadata

        in_path = os.path.join(args.storage, model_dir, FEATURES_PATH)

        for split_num in spl[args.silo]["splits"]:
            split, test_y = load_split_features(in_path, spl[args.silo], split_num)
            if split_num not in features:
                features[split_num] = {}
            for k in split:
                if k not in features[split_num]:
                    features[split_num][k] = []
                features[split_num][k].append(split[k])
    return features, old_metadata

def stack_features(args, model_dirs):
    features, old_metadata = load_features(args, model_dirs)
    if not features or old_metadata is None:
        print("Did not find any extracted features to combine into ensemble")
        return {}, {}

    new_metadata = make_metadata(args, old_metadata)
    model_name = metadata_hash(new_metadata)

    out_path = os.path.join(args.storage, args.out_dir, FEATURES_PATH)
    os.makedirs(out_path, exist_ok=True)

    new_features = {}
    for split_num in features:
        feats = {}
        for k, v in features[split_num].items():
            X = np.concatenate([model_x for model_x, model_y in v], axis=1)
            y = v[0][1]
            feats[k] = (X, y)
        filenames = save_model_split(model_name, args.silo, out_path, split_num, feats)
        new_features[split_num] = filenames

    return new_features, new_metadata

def make_metadata(args, old_metadata):
    metadata = {
        "experiment_id": old_metadata["experiment_id"],
        "mode": "ensemble",
        "train_silo": ",".join(args.model_dir),
        "subset_name": old_metadata["subset_name"],
        "steps": 0,
        "model_arch": old_metadata["model_arch"],
        "epsilon": old_metadata["epsilon"],
        "max_grad_norm": old_metadata["max_grad_norm"],
        "model_file": "",
        "split_num": old_metadata.get("split_num", -1)
    }
    return metadata

TEST_SILO = "G12EC"
#TEST_SILO = "Code15"
#TEST_SILO = "Hefei"

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", help="data and model files location",
            default=".", type=str)
    parser.add_argument("--model-dir", help="model subdirectories (relative to storage)",
            default=[], type=str, action="append")
    parser.add_argument("--out-dir", help="new model directory to create",
            default="", type=str)
    parser.add_argument("--silo", help="hospital to evaluate",
            default=TEST_SILO, type=str)
    args = parser.parse_args()

    if not args.model_dir:
        print("--model-dir parameter is mandatory")
        sys.exit(1)

    if not args.out_dir:
        print("--out-dir parameter is mandatory")
        sys.exit(1)

    print("using", args.storage, "for input")
    print("model directories", args.model_dir)
    new_splits, new_metadata = stack_features(args, args.model_dir)

    if not new_splits:
        sys.exit(2)

    log_path = os.path.join(args.storage, args.out_dir, LOG_DIR)
    os.makedirs(log_path, exist_ok=True)

    split_meta_file = os.path.join(log_path, SPLITS_FILE)
    splits_meta = get_meta_file(split_meta_file)

    splits_meta.update({args.silo: {
                "splits": new_splits,
                "silo_name": args.silo,
                "subset_name": new_metadata["subset_name"],
                "model_arch": new_metadata["model_arch"],
                "experiment_id": new_metadata["experiment_id"]
            }
        })
    with open(split_meta_file, "w") as f:
        json.dump(splits_meta, f, indent=2)
        print("split metadata written to", split_meta_file)
    with open(os.path.join(log_path, METADATA), "w") as f:
        json.dump(new_metadata, f, indent=2)

