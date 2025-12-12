# ECG federated learning with differential privacy 

Code to supplement the paper:

*Differentially Private Federated Learning in Multihospital ECG Classification: Empirical Evaluation* by Priit Järv, Chito Patiño, Andrei Kazlouski, Andrei Kazlouski, Zoher Orabe, Tapio Pahikkala and Antti Airola (submitted for publication)

Purpose: train, finetune and evaluate differentially private federated learning (DPFL) algorithms for 12-lead ECG multilabel classification.

Implemented algorithms:
- DP-FedAvg
- DP-FedProx
- DP-FedRep
- DP-FedAdam (central privacy model)
- MR-MTL (personalized FL)

Additionally, you can:
- train cross-silo DP models
- train single-silo DP models
- build ensemble DP models
- finetune models fast using feature extraction
 
We use OpenFL to simulate federated learning. This is experimental code and not designed to be a convenient software package, so dataset preparation must be done manually (instructions provided). Tested on 8 public ECG datasets (CinC 2021 challenge, Shandong provincial hospital, CODE-15%, Hefei challenge, MIMIC IV ECG).

## Setup

```
python3 -m venv openfl
. openfl/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -r openfl_requirements.txt
```

You can pick whatever name you want for the virtual environment, "openfl" is an example.

## Data

Data download links and preprocessing scripts:

| Dataset | Download | Credentialed access | Preprocessing | 
| ------ | ------ |----------------|----------------|
| ChapmanShaoxing & Ningbo | https://physionet.org/content/challenge-2021/1.0.3/#files       |        -        |  https://github.com/UTU-Health-Research/dl-ecg-classifier/blob/main/notebooks/1_setup_runs.ipynb |
| CPSC & CPSC-Extra | https://physionet.org/content/challenge-2021/1.0.3/#files       |        -        |  https://github.com/UTU-Health-Research/dl-ecg-classifier/blob/main/notebooks/1_setup_runs.ipynb |
| G12EC | https://physionet.org/content/challenge-2021/1.0.3/#files       |        -        |  https://github.com/UTU-Health-Research/dl-ecg-classifier/blob/main/notebooks/1_setup_runs.ipynb |
| PTB & PTBXL | https://physionet.org/content/challenge-2021/1.0.3/#files       |        -        |  https://github.com/UTU-Health-Research/dl-ecg-classifier/blob/main/notebooks/1_setup_runs.ipynb |
| SPH | https://doi.org/10.6084/m9.figshare.c.5779802 | - |  https://github.com/UTU-Health-Research/dl-ecg-classifier/blob/main/notebooks/1_setup_runs.ipynb |
| Code 15% | https://zenodo.org/records/4916206 | - | [code_preproc.ipynb](code_preproc.ipynb) |
| Hefei | https://tianchi.aliyun.com/competition/entrance/231754/ | - | [hefei_preproc.ipynb](hefei_preproc.ipynb) |
| MIMIC IV ECG | https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/ (labels), https://physionet.org/content/mimic-iv-ecg/1.0/ (ECG) | PhysioNet | [mimic_preproc.ipynb](mimic_preproc.ipynb) |

The preprocessed data should be in the `data/processed_data` and the metadata/label files in the `data/split_csvs` subdirectory, with the following structure (the data file counts may differ, some formats have 2 files per ECG and this tree snapshot was made from our local copy with some extra unused ECG files):

```
data
├── processed_data
│   ├── ChapmanShaoxing_Ningbo  [90304 entries]
│   ├── Code15  [345780 entries]
│   ├── Code15_test  [827 entries]
│   ├── CPSC_CPSC_Extra  [20660 entries]
│   ├── G12EC  [20688 entries]
│   ├── Hefei  [20036 entries]
│   ├── Hefei_compat -> Hefei
│   ├── Mimic  [37909 entries]
│   ├── Mimic_test  [39343 entries]
│   ├── PTB_PTBXL  [44706 entries]
│   └── SPH  [23275 entries]
└── split_csvs
    ├── ChapmanShaoxing_Ningbo
    │   ├── clean_all_chapman_10_perc.csv
    │   └── clean_all_chapman.csv
    ├── Code15
    │   ├── all_ecgs.csv
    │   ├── ecgs_patients_04_percent.csv
    │   ├── ecgs_patients_20_percent.csv
    │   └── patient_ids.csv
    ├── Code15_test
    │   └── all_ecgs.csv
    ├── CPSC_CPSC_Extra
    │   ├── clean_all_cpsc_10_perc.csv
    │   └── clean_all_cpsc.csv
    ├── G12EC
    │   ├── clean_all_g12ec_10_perc.csv
    │   └── clean_all_g12ec.csv
    ├── Hefei
    │   ├── ecgs_32classes_10.csv
    │   ├── ecgs_32classes.csv
    │   ├── ecgs_with_fullmetadata_10.csv
    │   └── ecgs_with_fullmetadata.csv
    │   ├── ecgs_pretrained_compatible_10.csv
    │   └── ecgs_pretrained_compatible.csv
    ├── Mimic
    │   ├── ecgs_200classes_10.csv
    │   └── ecgs_200classes.csv
    ├── Mimic_test
    │   ├── ecgs_pretrained_compatible_10.csv
    │   └── ecgs_pretrained_compatible.csv
    ├── PTB_PTBXL
    │   ├── clean_all_ptb_10_perc.csv
    │   └── clean_all_ptb.csv
    └── SPH
        ├── clean_all_sph_10_perc.csv
        └── clean_all_sph.csv
```

The dataset files can be configured by editing `src/dataloader/silo.py`

## Command line scripts

The CLI scripts assume that each model is placed in their own subdirectory, that contains snapshots, statistics, metadata, extracted features, finetuned classifier heads and saved predictions. Most scripts require the `--model-dir` parameter.

The script `example.sh` is an example of training, finetuning and evaluating a DP-FedAvg model using 4 training silos and one target silo using the Python scripts.

You can run the Python scripts with the `--help` parameter to see the full set of command line options.

### DP training

The `dp_training.py` does not have a separate `--model-dir` parameter and is assumed to run from inside the model directory. However, you can still point it to the model directory using the `--checkpoints` and `--logs` parameters. The following command trains a DP-FedAvg model with epsilon 1.0, max gradient norm 0.7, and using the "v4" model architecture.

```
python dp_training.py --storage=. \
	--checkpoints=model/checkpoints --logs=model/logs \
	--mode=fedavg_local --experiment-id 0 \
	--train-silo=ChapmanShaoxing_Ningbo,SPH,PTB_PTBXL,CPSC_CPSC-Extra --subset-name=full --model-arch=v4 \
	--num-rounds=30 --n-collab=4 --local-epoch=2 --epsilon=10.0 --max-grad-norm=0.7 \
	--lr=0.002 --phys-batch-size=512
```

The "full" and "small" subsets are defined in `src/dataloader/silo.py`, the latter is for testing with 10% of the data.

Table of algorithms and extra parameters here

Table of model architectures

### Feature extraction

```
python feature_extract.py --storage=. \
	--model-dir=model --silo=Code15 --subset-name=full --extractor=classif
```

This passes the CODE-15% dataset through the trained model in `model` subdirectory and stores the extracted features under `model/ext_features`. This provides substantial speedup in finetuning. You can use "res2x" instead of "classif" to finetune the last two residual blocks, but this can backfire by overfitting to your finetuning data and requires a lot of disk space. By default, `feature_extract.py` assumes you want to do k-fold crossvalidation, so the features are stored as 5-fold CV splits with train, validation and test parts.

### Finetuning

```
python finetune.py --storage=. \
	--model-dir=model --silo=Code15  \
	--train-set=train --local-epoch=200
```

Finetune the model to the silo over 200 epochs. This stores the finetuned classifier under `model/finetune`. A model can be finetuned to several datasets at once, this is handled by metadata stored into the model directory. If you supply `--train-set=train_1K` the model is finetuned on a 1K subsample of the training part. If the features were extracted in CV mode (the default), this command finetunes a classifier for each split.

### Evaluation

```
python evaluate.py --storage . --model-dir=model \
		--train-set=train --silo=Code15
```

Evaluates a finetuned model using 5-fold CV. The `--train-set` and `--silo` parameters select the finetuned head to use. It is also possible to evaluate a full model on a full dataset without any feature extraction and without CV, by setting `--mode=full` and `--num-splits=1` from the command line:

```
python evaluate.py --storage=. \
	--model-dir=model --silo=Code15_test --subset-name=full \
	--mode full --num-splits=1
```

Both uses of the command also store model predictions under `model/predictions`. Evaluation results are written to `model/logs/evaluation.json`.

### Ensemble

You can evaluate models as an ensemble. This requires that you first run `evaluate.py` with all models you include into the ensemble.

```
python ensemble_evaluate.py --storage . --model-dir=model1 --model-dir=model2 --model-dir=model3 --model-dir=model4 \
                        --out-file=my_ensemble/logs/evaluation.json --silo=Code15_test
```

This also works with finetuned models:

```
python ensemble_evaluate.py --storage . --model-dir=model1 --model-dir=model2 --model-dir=model3 --model-dir=model4 \
                        --out-file=my_ensemble/logs/evaluation.json --silo=Code15 --train-set=train
```

### Other

`summary.py` prints a short summary of the evaluation results (see `summary.py --help`).

## Codebase functionality

The notebooks demonstrate training and finetuning DP models directly from Python code. This may be helpful if you are interested in using our implementations of DP algorithms.

* [Model training](ecg_fl.ipynb)
* [Finetuning the trained model](ecg_fl_finetune.ipynb)

## Reproducibility

Completing the experiments as presented in the paper took 3500 hours on a dedicated GPU cluster. Fully reproducing them requires serious commitment of computational resources. Therefore, we do not provide our site-specific scripts to run the full set of experiments. If necessary, the `example.sh` script can be used to run the DPFL external validation and finetuning experiments, by building simple automation around it.

## License

See `LICENSE.txt`.

The ResNet-SE PyTorch implementation was released under BSD 2 clause license, included in `LICENSE-ResNetSE.txt`. If you create derivative work that does not include the ResNet-SE model you may remove the BSD 2 clause license.
