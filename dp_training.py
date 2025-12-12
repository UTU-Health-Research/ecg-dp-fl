#!/usr/bin/env python

import os
import os.path
import torch
#import torch.nn as nn
#import torch.nn.functional as F
import numpy as np
import random
import copy
import json
import collections
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 7, 5

from src.modeling.models import seresnet18
from src.modeling.train_utils import trainer_setup, adamopt, sgdopt, fedproxopt
from src.dataloader.dataset import dataloader
from src.modeling.metrics import train_loss_progression, fl_train_progression
from src.dataloader.silo import get_silos, get_experiment_seed, dataset_splits, concat_silos, single_split

from src.federated.dpfl import DPTrainEmbed, DPTrainStep, FederatedFlow, FedProxFlow, MRMTLFlow, DPAdamFlow, fl_setup

RANDOM_SEED = 42

# defaults, can be overriden from command line
CHECKPOINT_PATH = "checkpoints"
LOG_DIR = "logs"
ECG_DIR = "data/processed_data"
CSV_DIR = "data/split_csvs"

#SUBSET_NAME = "small"
SUBSET_NAME = "full"

#LEARNING_RATE = 0.003
LEARNING_RATE = 0.001
IN_CHANNELS=12

# virtual batch size, used to determine sampling rate
DP_BATCH_SIZE=1024
#DP_BATCH_SIZE=512 # for testing with "small" subset
# actual batch size used for the model
PHYS_BATCH_SIZE=256

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device

def trainer_model_setup(train_silos, k, device, cp_path, optfunc=adamopt, model_arch="v4", privacy_params={}, experiment_id=0, batch_size=PHYS_BATCH_SIZE,
        split_num=-1, train_set="train", subset_name="full"):
    labels = train_silos[k].columns.tolist()[4:]
    seed = get_experiment_seed(k, experiment_id)

    if privacy_params:
        bs = DP_BATCH_SIZE
    else:
        bs = batch_size
    if split_num < 0:
        train_dl = dataloader(train_silos[k], device=device, batch_size=bs)
        val_dl = None
    else:
        if subset_name.startswith("strat"):
            splits = single_split(train_silos[k], None, random_state=seed)
        else:
            splits = dataset_splits(k, train_silos, random_state=seed)
        train_dl = dataloader(splits[train_set][split_num], train=True, device=device, batch_size=bs)
        val_dl = dataloader(splits["val"][split_num], train=False, device=device, batch_size=bs)

    cp_params = {"path": cp_path, "freq": 1}
    model = seresnet18.resnet18(seed, model_arch, in_channel=IN_CHANNELS, out_channel=len(labels))

    trainer = trainer_setup(model, train_dl, labels, device, optfunc,
            cp_params=cp_params, privacy_params=privacy_params, name=k, val_dl=val_dl)

    return trainer

#
# This needs to be re-run for each experiment, because local collaborators have persistent state
#
def fl_trainers_setup(train_silos, train_names, device, cp_path, optfunc=adamopt, model_arch="v4", privacy_params={}, experiment_id=0, trainer_class=None,
        batch_size=PHYS_BATCH_SIZE):
    labels = train_silos[train_names[0]].columns.tolist()[4:]

    trainers = {}
    total_size = 0
    for k in train_names:
        ds_size = train_silos[k].shape[0]
        total_size += ds_size
        if privacy_params:
            if "sample_rate" in privacy_params: # needed for classical DP-FedAvg
                bs = int(ds_size * privacy_params["sample_rate"])
            else:
                bs = DP_BATCH_SIZE
            train_dl = dataloader(train_silos[k], device=device, batch_size=bs)
        else:
            train_dl = dataloader(train_silos[k], device=device, batch_size=batch_size)
        cp_params = {"path": cp_path, "freq": None}
        seed = get_experiment_seed(k, experiment_id)
        model = seresnet18.resnet18(seed, model_arch, in_channel=IN_CHANNELS, out_channel=len(labels))

        trainers[k] = trainer_setup(model, train_dl, labels, device, optfunc,
                cp_params=cp_params, privacy_params=privacy_params, name=k, trainer_class=trainer_class)

    return trainers, total_size

def make_privacy_params(args, n_silos=1):
    if args.mode == "central" or n_silos == 1:
        # assume local or central modes
        silo_sample_rate = 1
        total_epochs = args.local_epoch
    else:
        assert args.n_collab <= n_silos
        silo_sample_rate = args.n_collab / n_silos
        total_epochs = args.local_epoch * args.num_rounds
    e_epochs = int(silo_sample_rate * total_epochs)

    if args.sigma > 0.0000001:
        privacy_params = {
            "epsilon": args.epsilon,
            "max_grad_norm": args.max_grad_norm,
            "phys_batch_size": args.phys_batch_size,
            "e_epochs": e_epochs,
            "sigma": args.sigma}
    elif args.epsilon > 0.0000001:
        privacy_params = {
            "epsilon": args.epsilon,
            "max_grad_norm": args.max_grad_norm,
            "phys_batch_size": args.phys_batch_size,
            "e_epochs": e_epochs}
    else:
        privacy_params = {}
    return privacy_params

def train_silo_names(args):
    return args.train_silo.split(",")

def run_mode_local(silos, train_silo, device, cp_path,
        privacy_params, optfunc, args):
    trainer = trainer_model_setup(silos, train_silo, device, cp_path,
        privacy_params=privacy_params,
        optfunc=optfunc,
        model_arch=args.model_arch,
        experiment_id=args.experiment_id,
        batch_size=args.phys_batch_size,
        split_num=args.split_num,
        train_set=args.train_set,
        subset_name=args.subset_name)

    hist = trainer.train(args.local_epoch)
    summary_stats = trainer.summary_stats()
    return hist, summary_stats, trainer.model_path

# train_names = ["ChapmanShaoxing_Ningbo", "CPSC_CPSC-Extra", "PTB_PTBXL", "SPH"]
def run_mode_federated(silos, train_names, device, cp_path,
        privacy_params, optfunc, args):
    train_silos = dict((k, v) for k, v in silos.items() if k in train_names)
    local_runtime = fl_setup(train_silos)

    if args.mode == "fedrep_local":
        print("algorithm: FedRep")
        trainer_class = DPTrainEmbed
        flow_class = FederatedFlow
    elif args.mode == "fedprox_local":
        print("algorithm: FedProx")
        trainer_class = None  # use default
        flow_class = FedProxFlow
    elif args.mode == "mrmtl_local":
        print("algorithm: MR-MTL")
        trainer_class = None  # use default
        flow_class = MRMTLFlow
    elif args.mode == "fedsgd_shared":
        print("algorithm: FedAdam")
        trainer_class = DPTrainStep
        flow_class = DPAdamFlow
        privacy_params["sample_rate"] = args.sample_rate
        privacy_params["sigma"] = 0.0 # testing only
    else:
        print("algorithm: FedAvg")
        trainer_class = None  # use default
        flow_class = FederatedFlow
    local_trainers, n_samples = fl_trainers_setup(train_silos, train_names, device, cp_path,
        privacy_params=privacy_params,
        optfunc=optfunc,
        model_arch=args.model_arch,
        experiment_id=args.experiment_id,
        trainer_class=trainer_class,
        batch_size=args.phys_batch_size)

    federated_flow = flow_class(num_rounds=args.num_rounds,
           local_epoch=args.local_epoch,
           n_collab=args.n_collab,
           cp_path=cp_path,
           trainer_log=True)
    if args.mode == "fedsgd_shared":
        federated_flow.eta = args.lr
        federated_flow.make_private(n_samples, privacy_params=privacy_params)
    federated_flow.runtime = local_runtime
    federated_flow.global_weights = local_trainers[train_names[0]].get_weights()
    federated_flow.get_trainer = lambda x: local_trainers[x]
    federated_flow.run()

    return federated_flow.stats, federated_flow.summary, federated_flow.model_path

def make_metadata(args, model_path):
    if type(model_path) == dict:
        model_file = dict((k, os.path.basename(v)) for k, v in model_path.items())
    else:
        model_file = os.path.basename(model_path)
    metadata = {
        "experiment_id": args.experiment_id,
        "mode": args.mode,
        "train_silo": args.train_silo,
        "subset_name": args.subset_name,
        "steps": args.local_epoch if args.mode in ["local", "central"] else args.num_rounds,
        "model_arch": args.model_arch,
        "epsilon": args.epsilon,
        "max_grad_norm": args.max_grad_norm,
        "model_file": model_file,
        "split_num": args.split_num
    }
    return metadata

if __name__ == "__main__":
    import argparse
    import time

    print("Script loaded at {}".format(time.ctime()))

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="training mode: local, central, fedavg_local, fedrep_local, fedsgd_shared",
            default="local", type=str)
    parser.add_argument("--experiment-id", help="experiment id (for random seed)",
            default=0, type=int)
    parser.add_argument("--storage", help="data and output files location",
            default=".", type=str)
    parser.add_argument("--checkpoints", help="checkpoints subdirectory (relative to storage)",
            default=CHECKPOINT_PATH, type=str)
    parser.add_argument("--logs", help="logs and graphs subdirectory (relative to storage)",
            default=LOG_DIR, type=str)
    parser.add_argument("--local-epoch", help="number of epochs each round",
            default=1, type=int)
    parser.add_argument("--num-rounds", help="number of federated rounds",
            default=5, type=int)
    parser.add_argument("--n-collab", help="number of collaborators each round",
            default=4, type=int)
    parser.add_argument("--train-silo", help="hospital to train on",
            default="G12EC", type=str)
    parser.add_argument("--subset-name", help="subset to train on (full/small)",
            default=SUBSET_NAME, type=str)
    parser.add_argument("--split-num", help="split number (with local mode only, train on one fold with validation)",
            default=-1, type=int)
    parser.add_argument("--train-set", help="train set name (train/train_1K), local mode with splits only",
            default="train", type=str)
    parser.add_argument("--model-arch", help="SEResNet18 model architecture v1-v4",
            default="v4", type=str)
    parser.add_argument("--epsilon", help="privacy budget (set to 0 to disable DP)",
            default=10.0, type=float)
    parser.add_argument("--max-grad-norm", help="gradient clipping parameter",
            default=1.0, type=float)
    parser.add_argument("--sigma", help="noise magnitude (if given, epsilon is ignored)",
            default=0.0, type=float)
    parser.add_argument("--optfunc", help="optimizer (adam|sgd)",
            default="adam", type=str)
    parser.add_argument("--lr", help="learning rate",
            default=LEARNING_RATE, type=float)
    parser.add_argument("--mu", help="regularization parameter (for fedprox_local)",
            default=0.01, type=float)
    parser.add_argument("--sample-rate", help="sample rate (for fedsgd_shared)",
            default=0.1, type=float)
    parser.add_argument("--phys-batch-size", help="batch size on GPU",
            default=PHYS_BATCH_SIZE, type=int)
    args = parser.parse_args()
    print("using", args.storage, "for input")

    device = get_device()

    silos, _ = get_silos(os.path.join(args.storage, CSV_DIR),
                          os.path.join(args.storage, ECG_DIR),
                          subset=args.subset_name)

    cp_path = os.path.join(args.storage, args.checkpoints)
    log_path = os.path.join(args.storage, args.logs)
    os.makedirs(cp_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    print("using", cp_path, "for checkpoints")
    print("using", log_path, "for logs and graphs")

    if args.optfunc == "sgd":
        optfunc = lambda x: sgdopt(x, lr=args.lr)
        print("using SGD optimizer")
    elif args.mode in ["fedprox_local", "mrmtl_local"]:
        optfunc = lambda x: fedproxopt(x, lr=args.lr, mu=args.mu)
        print("using FedProxAdam optimizer, mu={:.4f}".format(args.mu))
    else:
        optfunc = lambda x: adamopt(x, lr=args.lr)
        print("using Adam optimizer")

    train_silos = train_silo_names(args)
    privacy_params = make_privacy_params(args, len(train_silos))

    if args.mode == "local":
        hist, summary_stats, m_path = run_mode_local(silos, train_silos[0], device, cp_path,
                privacy_params, optfunc, args)
    elif args.mode == "central":
        silos["ALL"] = concat_silos(silos, train_silos)
        hist, summary_stats, m_path = run_mode_local(silos, "ALL", device, cp_path,
                privacy_params, optfunc, args)
    elif args.mode in ["fedavg_local", "fedrep_local", "fedprox_local", "mrmtl_local", "fedsgd_shared"]:
        fed_stats, summary_stats, m_path = run_mode_federated(silos, train_silos, device, cp_path,
                privacy_params, optfunc, args)
    else:
        raise ValueError("invalid mode")

    with open(os.path.join(log_path, "summary_stats.json"), "w") as f:
        json.dump(summary_stats, f, indent=2)
    with open(os.path.join(log_path, "metadata.json"), "w") as f:
        json.dump(make_metadata(args, m_path), f, indent=2)
    if args.mode in ["local", "central"]:
        with open(os.path.join(log_path, "train_history.json"), "w") as f:
            json.dump(hist, f, indent=2)
        train_loss_progression(hist, plot_dir=log_path)
    else: # federated modes
        with open(os.path.join(log_path, "fl_train_history.json"), "w") as f:
            json.dump(fed_stats, f, indent=2)
        fl_train_progression(fed_stats, plot_dir=log_path, key="train_loss", label="Training loss", guides=[0.2, 0.3])
        fl_train_progression(fed_stats, plot_dir=log_path, key="train_macro_auroc", label="macro AUC", guides=[])

