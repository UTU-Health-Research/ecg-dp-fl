#!/usr/bin/env python
# coding: utf-8

# # Setup

import os
import os.path
import pandas as pd
import numpy as np
import random
import collections
import json
from sklearn.model_selection import KFold, train_test_split
import warnings


HOSP_NAMES = ["ChapmanShaoxing_Ningbo", "CPSC_CPSC-Extra", "G12EC", "PTB_PTBXL", "SPH",
        "Code15", "Code15_test", "Hefei", "Hefei_compat", "Mimic", "Mimic_test"]
ECG_DIR = "data/processed_data"
CSV_DIR = "data/split_csvs"

#SUBSET_NAME = "small"
SUBSET_NAME = "full"
SUBSET_CONF = {
    "small" : {
        "ChapmanShaoxing_Ningbo": "clean_all_chapman_10_perc.csv",
        "CPSC_CPSC-Extra": "clean_all_cpsc_10_perc.csv",
        "G12EC": "clean_all_g12ec_10_perc.csv",
        "PTB_PTBXL": "clean_all_ptb_10_perc.csv",
        "SPH": "clean_all_sph_10_perc.csv",
        "Code15": "ecgs_patients_04_percent.csv",
        "Code15_test": "all_ecgs.csv",
        "Hefei": "ecgs_32classes_10.csv",
        "Hefei_compat": "ecgs_pretrained_compatible_10.csv",
        "Mimic": "ecgs_200classes_10.csv",
        "Mimic_test": "ecgs_pretrained_compatible_10.csv"
    },
    "full" : {
        "ChapmanShaoxing_Ningbo": "clean_all_chapman.csv",
        "CPSC_CPSC-Extra": "clean_all_cpsc.csv",
        "G12EC": "clean_all_g12ec.csv",
        "PTB_PTBXL": "clean_all_ptb.csv",
        "SPH": "clean_all_sph.csv",
        "Code15": "ecgs_patients_20_percent.csv",
        "Code15_test": "all_ecgs.csv",
        "Hefei": "ecgs_32classes.csv",
        "Hefei_compat": "ecgs_pretrained_compatible.csv",
        "Mimic": "ecgs_200classes.csv",
        "Mimic_test": "ecgs_pretrained_compatible.csv"
    },
    "strat0" : {
        "ChapmanShaoxing_Ningbo": "stratified0_train.csv",
        "CPSC_CPSC-Extra": "stratified0_train.csv",
        "PTB_PTBXL": "stratified0_train.csv",
        "SPH": "stratified0_train.csv"
    },
    "strat0_test" : {
        "ChapmanShaoxing_Ningbo": "stratified0_test.csv",
        "CPSC_CPSC-Extra": "stratified0_test.csv",
        "PTB_PTBXL": "stratified0_test.csv",
        "SPH": "stratified0_test.csv"
    },
    "strat1" : {
        "ChapmanShaoxing_Ningbo": "stratified1_train.csv",
        "CPSC_CPSC-Extra": "stratified1_train.csv",
        "PTB_PTBXL": "stratified1_train.csv",
        "SPH": "stratified1_train.csv"
    },
    "strat1_test" : {
        "ChapmanShaoxing_Ningbo": "stratified1_test.csv",
        "CPSC_CPSC-Extra": "stratified1_test.csv",
        "PTB_PTBXL": "stratified1_test.csv",
        "SPH": "stratified1_test.csv"
    },
    "strat2" : {
        "ChapmanShaoxing_Ningbo": "stratified2_train.csv",
        "CPSC_CPSC-Extra": "stratified2_train.csv",
        "PTB_PTBXL": "stratified2_train.csv",
        "SPH": "stratified2_train.csv"
    },
    "strat2_test" : {
        "ChapmanShaoxing_Ningbo": "stratified2_test.csv",
        "CPSC_CPSC-Extra": "stratified2_test.csv",
        "PTB_PTBXL": "stratified2_test.csv",
        "SPH": "stratified2_test.csv"
    }
}

OLD_PREFIX = "data/processed_data"
def patch_filenames(df, new_prefix):
    new_names = []
    for old_name in df["path"].tolist():
        new_names.append(old_name.replace(OLD_PREFIX, new_prefix))
    df = df.drop("path", axis=1)
    df.insert(0, "path", pd.Series(new_names))
    return df

def get_silo_df(hosp, path, ecg_path, subset=SUBSET_NAME):
    if hosp not in SUBSET_CONF[subset]:
        return None  # some subset/hospital combos are not needed
    fn = os.path.join(path, hosp.replace("-", "_"), SUBSET_CONF[subset][hosp])
    if not os.path.isfile(fn):
        print(f"skip loading silo {hosp}, csv not found")
        return None
    df = pd.read_csv(fn)
    return patch_filenames(df, ecg_path)

def get_silos(csv_path=CSV_DIR, ecg_path=ECG_DIR, subset=SUBSET_NAME):
    silos = {}
    for k in HOSP_NAMES:
        df = get_silo_df(k, csv_path, ecg_path, subset)
        if df is not None:
            silos[k] = df
    cinc_labels = silos[HOSP_NAMES[0]].columns.tolist()[4:]
    return silos, cinc_labels

def concat_silos(silos, silo_names):
    df = pd.concat([v for k, v in silos.items() if k in silo_names],
            ignore_index=True)
    return df

def single_split(df_train, df_test, random_state=42):
    train_index, val_index = train_test_split(np.arange(df_train.shape[0]),
            test_size=0.1, random_state=random_state)
    train_splits = [df_train.iloc[train_index]]
    val_splits = [df_train.iloc[val_index]]
    train_1K = [train_splits[0].sample(1024)]
    test_splits = [df_test]
    return {"train": train_splits,
            "train_1K": train_1K,
            "val": val_splits,
            "test": test_splits}

def dataset_splits(silo_name, silos, k=5, random_state=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    train_splits = []
    val_splits = []
    test_splits = []
    for train_index, test_index in kf.split(silos[silo_name]):
        train_index, val_index = train_test_split(train_index, test_size=0.1, random_state=random_state)
        train_splits.append(silos[silo_name].iloc[train_index])
        val_splits.append(silos[silo_name].iloc[val_index])
        test_splits.append(silos[silo_name].iloc[test_index])
    train_1K = []
    for df in train_splits:
        # in some special cases
        sample_size = min(1024, df.shape[0])
        train_1K.append(df.sample(sample_size))
    return {"train": train_splits,
            "train_1K": train_1K,
            "val": val_splits,
            "test": test_splits}

def get_experiment_seed(hosp, experiment_id=0):
    seeds = EXPERIMENT_SEEDS.get(hosp, [])
    if experiment_id >= len(seeds):
        warnings.warn("using generic seed")
        return 42
    return seeds[experiment_id]

EXPERIMENT_SEEDS = {
    "ChapmanShaoxing_Ningbo": [ 
        662418406, 842400004, 524141987, 406802186, 828248992,
        561925047, 414218785, 293876101, 599828680, 964447679],
    "CPSC_CPSC-Extra": [
        10974390, 593621120, 758283498, 873841866, 526238104,
        968709248, 96365599, 758235525, 582523052, 589097371],
    "G12EC": [
        444726346, 289150538, 566852692, 319024325, 45878175,
        90245458, 626719924, 398925033, 285922445, 478970763],
    "PTB_PTBXL": [
        71810256, 707412047, 885440834, 842450468, 474293056,
        836585258, 333316916, 227185755, 671964215, 495154653],
    "SPH": [
        182173961, 392306216, 233456884, 219669417, 757462646,
        945348829, 544024921, 898284927, 727211822, 404371557],
    "Code15": [
        926713154, 190702860, 68561724, 391251194, 736145876,
        802749389, 951489349, 653555273, 547371962, 87294154],
    "Code15_test": [
        723181497, 660852527, 407222215, 328766223, 831034285,
        471642061, 483989588, 940033503, 947383996, 766090546],
    "Hefei": [
        998122118, 257268063, 530105041, 499812500, 77971983,
        457348778, 470379996, 518047556, 372903033, 626461382],
    "Hefei_compat": [
        881941934, 997498964, 911382647, 301971587, 891175121,
        95040810, 464178732, 69008973, 646725270, 682363370],
    "Mimic": [
        271282730, 532024088, 843171888, 564362504, 678835117,
        820529910, 119656769, 480358316, 625498622, 893134635],
    "Mimic_test": [
        478771941, 177065408, 745633213, 10723519, 832785444,
        82110961, 669217257, 346293010, 588108160, 744426061],
    "ALL": [  # "virtual" silo, generated on the fly
        489976184, 983932931, 234736494, 123937374, 399973448,
        981226988, 526098077, 116211577, 923393129, 855296358]
}

# "True" random numbers, use them if more experiment seeds are needed
RANDOM_POOL = [
    682518509, 907155041, 522203693, 401004104, 357998652,
    7470582, 250404686, 641533588, 681178597, 32798244,
    774109525, 813847916, 651365891, 289500095, 935187600,
    226997867, 258551585, 497974451, 731652058, 856658875,
    442290084, 627849838, 2286024, 283396159, 861097395,
    313516109, 714307783, 577417972, 349702626, 615054350,
    192209782, 588552187, 624623205, 824698309, 450039994,
    543582545, 853378476, 201793485, 971285333, 198028621,
    677509700, 233570480, 827806254, 118184262, 774286181,
    162000151, 175548837, 240313146, 902783156, 836989077,
    742304709, 210639881, 55250222, 717612357, 803118360,
    366612156, 220681218, 448030337, 767485182, 382098747,
    130805389, 280435000, 847703278, 596756322, 891770355,
    281705708, 501995922, 852664975, 979690446, 690554543,
    607833736, 112488648, 357298684, 16935385, 743789944,
    667055396, 621051987, 613133616, 614852534, 170815862
]
