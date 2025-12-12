#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import json
import numpy as np

# relative to model path (hardcoded)
LOG_DIR = "logs"

def get_meta_file(fn):
    metadata = {}
    if os.path.isfile(fn):
        with open(fn) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    return metadata

def report(eval_data, args):
    if args.silo in eval_data:
        out = [args.model_dir, args.silo]
        for k, v in eval_data[args.silo]["splits"]["train"].items():
            out.append("{} {:.3f}".format(k, np.mean(list(v.values()))))
        print(" ".join(out))
    else:
        print("Silo {args.silo} not found")

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
    parser.add_argument("--silo", help="hospital to evaluate",
            default=TEST_SILO, type=str)
    args = parser.parse_args()

    if not args.model_dir:
        print("--model-dir parameter is mandatory")
        sys.exit(1)

    model_dir = os.path.join(args.storage, args.model_dir)
    evals_file = os.path.join(model_dir, LOG_DIR, "evaluation.json")
    evals_summary = get_meta_file(evals_file)
    report(evals_summary, args)

