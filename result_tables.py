#!/usr/bin/env python
# coding: utf-8

import os
import os.path
import json
import numpy as np
import scipy.stats as stats

REPORT = "evaluation.json"
EPSILONS = ["1", "10"]

FT_LOCAL = {"label": "local models",
     "splits": {"train":
        {"Code15": {
                "0": "localtest_resnetv3_Code15_0_nondp",
                "1": "localtest_resnetv3_Code15_1_nondp",
                "2": "localtest_resnetv3_Code15_2_nondp",
                "3": "localtest_resnetv3_Code15_3_nondp",
                "4": "localtest_resnetv3_Code15_4_nondp"
            },
            "Mimic": {
                "0": "localtest_resnetv3_Mimic_0_nondp",
                "1": "localtest_resnetv3_Mimic_1_nondp",
                "2": "localtest_resnetv3_Mimic_2_nondp",
                "3": "localtest_resnetv3_Mimic_3_nondp",
                "4": "localtest_resnetv3_Mimic_4_nondp"
            },
            "Hefei": {
                "0": "localtest_resnetv3_Hefei_0_nondp",
                "1": "localtest_resnetv3_Hefei_1_nondp",
                "2": "localtest_resnetv3_Hefei_2_nondp",
                "3": "localtest_resnetv3_Hefei_3_nondp",
                "4": "localtest_resnetv3_Hefei_4_nondp"
            }
        },
    "train_1K":
        {"Code15": {
                "0": "localtest_resnetv3_Code15_0_nondp_train_1K",
                "1": "localtest_resnetv3_Code15_1_nondp_train_1K",
                "2": "localtest_resnetv3_Code15_2_nondp_train_1K",
                "3": "localtest_resnetv3_Code15_3_nondp_train_1K",
                "4": "localtest_resnetv3_Code15_4_nondp_train_1K"
            },
            "Mimic": {
                "0": "localtest_resnetv3_Mimic_0_nondp_train_1K",
                "1": "localtest_resnetv3_Mimic_1_nondp_train_1K",
                "2": "localtest_resnetv3_Mimic_2_nondp_train_1K",
                "3": "localtest_resnetv3_Mimic_3_nondp_train_1K",
                "4": "localtest_resnetv3_Mimic_4_nondp_train_1K"
            },
            "Hefei": {
                "0": "localtest_resnetv3_Hefei_0_nondp_train_1K",
                "1": "localtest_resnetv3_Hefei_1_nondp_train_1K",
                "2": "localtest_resnetv3_Hefei_2_nondp_train_1K",
                "3": "localtest_resnetv3_Hefei_3_nondp_train_1K",
                "4": "localtest_resnetv3_Hefei_4_nondp_train_1K"
            }
        }
    }
}

FULL_NONDP = {
    "label": "nondp baseline",
     "models": {"1": "central_resnetv3_full_nondp",
                "10": "central_resnetv3_full_nondp"}
}

FT_REPORTS = [
    {
        "descr": "Finetuning: threat model 1",
        "rows": [
            {"label": "DP-FedAvg",
             "models": {"1": "fedavg_local_resnetv3_full_eps1",
                        "10": "fedavg_local_resnetv3_full_eps10"}
            },
            {"label": "DP-FedRep",
             "models": {"1": "fedrep_local_resnetv3_full_eps1",
                        "10": "fedrep_local_resnetv3_full_eps10"}
            },
            {"label": "DP-FedProx",
             "models": {"1": "fedprox_local_resnetv3_full_eps1",
                        "10": "fedprox_local_resnetv3_full_eps10"}
            },
            {"label": "DP ensemble",
             "models": {"1": "ensemble_eps1",
                        "10": "ensemble_eps10"}
            },
            FULL_NONDP,
            FT_LOCAL
        ],
        "train_set": "train",
        "test_sets": ["Code15", "Mimic", "Hefei"]
    },
    {
        "descr": "Finetuning small training set: threat model 1",
        "rows": [
            {"label": "DP-FedAvg",
             "models": {"1": "fedavg_local_resnetv3_full_eps1",
                        "10": "fedavg_local_resnetv3_full_eps10"}
            },
            {"label": "DP-FedRep",
             "models": {"1": "fedrep_local_resnetv3_full_eps1",
                        "10": "fedrep_local_resnetv3_full_eps10"}
            },
            {"label": "DP-FedProx",
             "models": {"1": "fedprox_local_resnetv3_full_eps1",
                        "10": "fedprox_local_resnetv3_full_eps10"}
            },
            {"label": "DP ensemble",
             "models": {"1": "ensemble_eps1",
                        "10": "ensemble_eps10"}
            },
            FULL_NONDP,
            FT_LOCAL
        ],
        "train_set": "train_1K",
        "test_sets": ["Code15", "Mimic", "Hefei"]
    },
    {
        "descr": "Finetuning: threat model 2a, 2b",
        "rows": [
            {"label": "DP-FedAdam",
             "models": {"1": "fedadam_shared_resnetv3_full_eps1",
                        "10": "fedadam_shared_resnetv3_full_eps10"}
            },
            {"label": "DP central",
             "models": {"1": "central_resnetv3_full_eps1",
                        "10": "central_resnetv3_full_eps10"}
            },
            FULL_NONDP,
            FT_LOCAL
        ],
        "train_set": "train",
        "test_sets": ["Code15", "Mimic", "Hefei"]
    },
    {
        "descr": "Finetuning small training set: threat model 2a, 2b",
        "rows": [
            {"label": "DP-FedAdam",
             "models": {"1": "fedadam_shared_resnetv3_full_eps1",
                        "10": "fedadam_shared_resnetv3_full_eps10"}
            },
            {"label": "DP central",
             "models": {"1": "central_resnetv3_full_eps1",
                        "10": "central_resnetv3_full_eps10"}
            },
            FULL_NONDP,
            FT_LOCAL
        ],
        "train_set": "train_1K",
        "test_sets": ["Code15", "Mimic", "Hefei"]
    },
]

PT_REPORTS = [
    {
        "descr": "Pre-training: threat model 1",
        "rows": [
            {"label": "DP-FedAvg",
             "models": {"1": "fedavg_local_resnetv3_full_eps1",
                        "10": "fedavg_local_resnetv3_full_eps10"}
            },
            {"label": "DP-FedProx",
             "models": {"1": "fedprox_local_resnetv3_full_eps1",
                        "10": "fedprox_local_resnetv3_full_eps10"}
            },
            {"label": "DP ensemble",
             "models": {"1": "ensemble_eps1",
                        "10": "ensemble_eps10"}
            },
            FULL_NONDP,
        ],
        "test_sets": ["Code15_test", "Mimic_test", "Hefei_compat"]
    },
    {
        "descr": "Pre-training: threat model 2a, 2b",
        "rows": [
            {"label": "DP-FedAdam",
             "models": {"1": "fedadam_shared_resnetv3_full_eps1",
                        "10": "fedadam_shared_resnetv3_full_eps10"}
            },
            {"label": "DP central",
             "models": {"1": "central_resnetv3_full_eps1",
                        "10": "central_resnetv3_full_eps10"}
            },
            FULL_NONDP,
        ],
        "test_sets": ["Code15_test", "Mimic_test", "Hefei_compat"]
    },
]

STRAT_LOCAL = {"label": "local models",
    "splits": {
        "ChapmanShaoxing_Ningbo": ["local_resnetv3_ChapmanShaoxing_Ningbo_strat0_nondp",
            "local_resnetv3_ChapmanShaoxing_Ningbo_strat1_nondp",
            "local_resnetv3_ChapmanShaoxing_Ningbo_strat2_nondp"],
        "CPSC_CPSC-Extra": ["local_resnetv3_CPSC_CPSC-Extra_strat0_nondp",
            "local_resnetv3_CPSC_CPSC-Extra_strat1_nondp",
            "local_resnetv3_CPSC_CPSC-Extra_strat2_nondp"],
        "PTB_PTBXL": ["local_resnetv3_PTB_PTBXL_strat0_nondp",
            "local_resnetv3_PTB_PTBXL_strat1_nondp",
            "local_resnetv3_PTB_PTBXL_strat2_nondp"],
        "SPH": ["local_resnetv3_SPH_strat0_nondp",
            "local_resnetv3_SPH_strat1_nondp",
            "local_resnetv3_SPH_strat2_nondp"]
    }
}

STRAT_NONDP = {
    "label": "nondp baseline",
    "models": {"1": ["central_resnetv3_strat0_nondp", "central_resnetv3_strat1_nondp", "central_resnetv3_strat2_nondp"],
                "10": ["central_resnetv3_strat0_nondp", "central_resnetv3_strat1_nondp", "central_resnetv3_strat2_nondp"]}
}

STRAT_REPORTS = [
    {
        "descr": "Federated members: threat model 1",
        "rows": [
            {"label": "DP-FedAvg",
             "models": {"1": ["fedavg_local_resnetv3_strat0_eps1", "fedavg_local_resnetv3_strat1_eps1", "fedavg_local_resnetv3_strat2_eps1"],
                        "10": ["fedavg_local_resnetv3_strat0_eps10", "fedavg_local_resnetv3_strat1_eps10", "fedavg_local_resnetv3_strat2_eps10"]}
            },
            {"label": "DP-FedRep",
             "models": {"1": ["fedrep_local_resnetv3_strat0_eps1", "fedrep_local_resnetv3_strat1_eps1", "fedrep_local_resnetv3_strat2_eps1"],
                        "10": ["fedrep_local_resnetv3_strat0_eps10", "fedrep_local_resnetv3_strat1_eps10", "fedrep_local_resnetv3_strat2_eps10"]}
            },
            {"label": "DP-FedProx",
             "models": {"1": ["fedprox_local_resnetv3_strat0_eps1", "fedprox_local_resnetv3_strat1_eps1", "fedprox_local_resnetv3_strat2_eps1"],
                        "10": ["fedprox_local_resnetv3_strat0_eps10", "fedprox_local_resnetv3_strat1_eps10", "fedprox_local_resnetv3_strat2_eps10"]}
            },
            {"label": "MR-MTL",
             "models": {"1": ["mrmtl_local_resnetv3_strat0_eps1", "mrmtl_local_resnetv3_strat1_eps1", "mrmtl_local_resnetv3_strat2_eps1"],
                        "10": ["mrmtl_local_resnetv3_strat0_eps10", "mrmtl_local_resnetv3_strat1_eps10", "mrmtl_local_resnetv3_strat2_eps10"]}
            },
            STRAT_NONDP,
            STRAT_LOCAL,
        ],
        "test_sets": ["ChapmanShaoxing_Ningbo", "CPSC_CPSC-Extra", "PTB_PTBXL", "SPH"]
    },
    {
        "descr": "Federated members: threat model 2a",
        "rows": [
            {"label": "DP-FedAdam",
             "models": {"1": ["fedadam_shared_resnetv3_strat0_eps1", "fedadam_shared_resnetv3_strat1_eps1", "fedadam_shared_resnetv3_strat2_eps1"],
                        "10": ["fedadam_shared_resnetv3_strat0_eps10", "fedadam_shared_resnetv3_strat1_eps10", "fedadam_shared_resnetv3_strat2_eps10"]}
            },
            STRAT_NONDP,
            STRAT_LOCAL,
        ],
        "test_sets": ["ChapmanShaoxing_Ningbo", "CPSC_CPSC-Extra", "PTB_PTBXL", "SPH"]
    },
]

def get_one_measurement(model_dir, test_sets, train_set=None, strategy=None, split_num=None, prefixes=["."]):
    test_results = {}
    for pref in prefixes:
        with open(os.path.join(pref, model_dir, REPORT)) as f:
            report = json.load(f)
            for test_set in test_sets:
                if test_set not in test_results:
                    test_results[test_set] = []
                if train_set is None:
                    spl = report[test_set]["splits"]
                else:
                    spl = report[test_set]["splits"][train_set]

                #print(spl)
                if split_num is None:
                    v = list(spl[strategy].values())
                else:
                    v = [spl[strategy][split_num]]
                test_results[test_set] += v
    return test_results

def iter_splits(row, test_set, train_set=None, strategy="classif"):
    if "splits" in row:
        if train_set is None:
            for model_dir in row["splits"][test_set]:
                yield "0", model_dir, "nondp", "pretrained", None
        elif train_set in row["splits"]:
            for split_num, model_dir in row["splits"][train_set][test_set].items():
                yield split_num, model_dir, "nondp", "pretrained", None
        else:
            raise KeyError("cannot find the training set in report configuration")
    else:
        for eps, model_dir in row["models"].items():
            if type(model_dir) == list:
                for md in model_dir:
                    yield None, md, eps, strategy, "train"
            else:
                yield None, model_dir, eps, strategy, train_set

def report(report_config, strategy="classif", prefixes=["."]):
    report = {"metadata": {
            "descr": report_config["descr"],
            "strategy": strategy,
            "test_sets": report_config["test_sets"]},
            "rows" : []
        }
    train_set = report_config.get("train_set")
    for row in report_config["rows"]:
        row_res = {"label": row["label"],
                "results": {}}
        for test_set in report_config["test_sets"]:
            all_v = {}
            for split_num, model_dir, eps, strat, ts in iter_splits(row, test_set, train_set, strategy):
                #print(split_num, model_dir, eps, strat)
                results = get_one_measurement(model_dir,
                                    [test_set],
                                    ts,
                                    strategy=strat,
                                    split_num=split_num,
                                    prefixes=prefixes)
                if eps not in all_v:
                    all_v[eps] = []
                all_v[eps] += results[test_set]
            for eps, v in all_v.items():
                if eps not in row_res["results"]:
                    row_res["results"][eps] = {}
                row_res["results"][eps][test_set] = v
        report["rows"].append(row_res)
    return report

def statistics(v):
    m, s, n = np.mean(v), np.std(v, ddof=1), len(v)
    t = stats.t.ppf(0.975, df=n - 1)
    e = t * (s / np.sqrt(n))
    return m, e

def fmt_val(mean, ci, compact=False):
    if compact:
        return "{:.3f}".format(mean)
    else:
        return "{:.3f}+-{:.3f}".format(mean, ci)

def fmt_tab(report, compact=False, separator="\t"):
    out = []
    out.append("{} {}".format(report["metadata"]["descr"], report["metadata"]["strategy"]))
    test_row = [""]
    eps_row = [""]
    for test_set in report["metadata"]["test_sets"]:
        test_row += [test_set, ""]
        eps_row += EPSILONS
    out.append(separator.join(test_row))
    out.append(separator.join(eps_row))

    for row in report["rows"]:
        row_res = [row["label"]]
        if "nondp" in row["results"]:
            all_eps = ["nondp"]
        else:
            all_eps = EPSILONS
        for test_set in report["metadata"]["test_sets"]:
            for eps in all_eps:
                v = row["results"][eps][test_set]
                mean, ci = statistics(v)
                v_str = fmt_val(mean, ci, compact)
                row_res.append(v_str)
                if eps == "nondp":
                    row_res.append(v_str)
        out.append(separator.join(row_res))
    return "\n".join(out)

def format_output(reports, fmt):
    output = ""
    if fmt == "json":
        output = json.dumps(reports, indent=2)
    elif fmt == "tab":
        for report in reports:
            output += "\n"  + fmt_tab(report)
    elif fmt == "tabcompact":
        for report in reports:
            output += "\n"  + fmt_tab(report, compact=True)
    return output

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", help="part of report (finetuning/pretrained/federated)",
            default="finetuning", type=str)
    parser.add_argument("--strategy", help="finetuning strategy (classif/res2x)",
            default="classif", type=str)
    parser.add_argument("--format", help="table format (tab/tabcompact/json)",
            default="tabcompact", type=str)
    parser.add_argument('filenames', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    reports = []
    if args.report == "finetuning":
        for rep in FT_REPORTS:
            reports.append(report(rep, strategy=args.strategy, prefixes=args.filenames))

    if args.report == "pretrained":
        for rep in PT_REPORTS:
            reports.append(report(rep, strategy="pretrained", prefixes=args.filenames))

    if args.report == "federated":
        for rep in STRAT_REPORTS:
            reports.append(report(rep, strategy=args.strategy, prefixes=args.filenames))

    print(format_output(reports, args.format))

