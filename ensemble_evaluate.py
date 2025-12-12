#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import numpy as np
import json

from src.modeling.metrics import cal_multilabel_metrics

LOG_DIR = "logs"
PRED_PATH = "predictions"

def gen_preds(mode, pred_logits):
    if mode == "mean":
        return np.mean(np.stack(pred_logits), axis=0)
    elif mode == "PoE":
        logits = np.stack(pred_logits)
        p = 1.0 / (1.0 + np.exp(-logits))
        poe = np.prod(p, axis=0) / (np.prod(p, axis=0) + np.prod(1 - p, axis=0))
        return poe
    else:
        raise NotImplementedError("invalid mode {}".format(mode))

def get_meta_file(fn):
    metadata = {}
    if os.path.isfile(fn):
        with open(fn) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    return metadata

def get_model_logits(args, model_dir):
    model_base = os.path.join(args.storage, model_dir)

    evals_file = os.path.join(model_base, LOG_DIR, "evaluation.json")
    evals_summary = get_meta_file(evals_file)
    if not args.silo in evals_summary:
        print(f"{model_dir} has not predictions for {args.silo}")
        return None

    if evals_summary[args.silo]["mode"] == "full":
        preds_set = evals_summary[args.silo]["predictions"]
    else:
        preds_set = evals_summary[args.silo]["predictions"][args.train_set]
    out = {}
    for k, v in preds_set.items():
        out[k] = {}
        for split_num, filename in v.items():
            preds_path = os.path.join(model_base, PRED_PATH, args.silo, filename)
            saved_data = np.load(preds_path)
            if k == "y_true":
                out[k][split_num] = saved_data["y_true"]
            else:
                out[k][split_num] = saved_data["y_hat"] # actually logits
    return out

def load_logits(args, model_dirs):
    all_preds = {}
    y_true = None
    for model_dir in model_dirs:
        model_data = get_model_logits(args, model_dir)
        if model_data is None:
            continue
        for k, v in model_data.items():
            if k == "y_true":
                if y_true is None:
                    y_true = v
                continue
            if k not in all_preds:
                all_preds[k] = {}
            for split_num, vv in v.items():
                if split_num not in all_preds[k]:
                    all_preds[k][split_num] = []
                all_preds[k][split_num].append(vv)
    return all_preds, y_true

def eval_ensemble(args, model_dirs):
    all_preds, y_true = load_logits(args, model_dirs)
    if not all_preds or y_true is None:
        print("Did not find any model predictions for computing ensemble predictions")
        return {}
    dummy_labels = ["l{}".format(i) for i in range(y_true["0"].shape[-1])]
    out = {}
    for k, v in all_preds.items():
        out[k] = {}
        print(args.silo, k)
        for split_num, pred_logits in v.items():
            y_hat = gen_preds(args.mode, pred_logits)
            _, _, test_macro_auroc, test_micro_auroc = cal_multilabel_metrics(y_true[split_num], y_hat, dummy_labels, 0.5)
            print(f"  {split_num} macro AUC {test_macro_auroc:.3f}")
            out[k][split_num] = test_macro_auroc
    return out

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
            default=[], type=str, action="append")
    parser.add_argument("--out-file", help="output JSON file",
            default="", type=str)
    parser.add_argument("--mode", help="calculation method (PoE|mean)",
            default="mean", type=str)
    parser.add_argument("--silo", help="hospital to evaluate",
            default=TEST_SILO, type=str)
    parser.add_argument("--train-set", help="train set name (train/train_1K), for finetuned models",
            default="full", type=str)
    args = parser.parse_args()

    if not args.model_dir:
        print("--model-dir parameter is mandatory")
        sys.exit(1)

    print("using", args.storage, "for input")
    new_evals = eval_ensemble(args, args.model_dir)

    if args.out_file:
        out_data = get_meta_file(args.out_file)
        if args.train_set == "full":
            # XXX: must explicitly set --train-set for finetuned models
            # technically, model metadata has this info but since
            # this script has multiple input models enforcing the
            # correct mode from command line is less error-prone
            curr_evals = new_evals
        else:
            curr_evals = out_data.get(args.silo, {}).get("splits", {})
            curr_evals[args.train_set] = new_evals
        out_data.update({args.silo: {
                    "splits": curr_evals,
                    "silo_name": args.silo,
                    "mode": "ensemble"
                }
            })
        with open(args.out_file, "w") as f:
            json.dump(out_data, f, indent=2)
            print("ensemble evaluation saved to", args.out_file)

