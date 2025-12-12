#!/bin/bash

# directory setup
# RUN_DIR should contain data/ directory hierarchy with all datasets, see README.md
RUN_DIR=${HOME}/ecg-dp-fl
CHECKPOINT_DIR=checkpoints
LOG_DIR=logs
MODEL_DIR=model
STORAGE=.
RESULTS_DIR=.
PYTHON_CMD="python"

# the commented out parameters make training much longer and the model much better
# if you use the small subset, edit the value of DP_BATCH_SIZE in dp_training.py
SUBSET_NAME="small"
#SUBSET_NAME="full"
NUM_ROUNDS=10
#NUM_ROUNDS=50
MODEL_ARCH="v4"
#MODEL_ARCH="v3"

mkdir -p ${STORAGE}/${MODEL_DIR}/${CHECKPOINT_DIR}
mkdir -p ${STORAGE}/${MODEL_DIR}/${LOG_DIR}

cd ${RUN_DIR}
${PYTHON_CMD} dp_training.py --storage=${STORAGE} \
	--checkpoints=${STORAGE}/${MODEL_DIR}/${CHECKPOINT_DIR} --logs=${STORAGE}/${MODEL_DIR}/${LOG_DIR} \
	--mode=fedavg_local --experiment-id 5 \
	--train-silo=ChapmanShaoxing_Ningbo,SPH,PTB_PTBXL,CPSC_CPSC-Extra --subset-name=${SUBSET_NAME} --model-arch=${MODEL_ARCH} \
	--num-rounds=${NUM_ROUNDS} --n-collab=4 --local-epoch=2 --epsilon=10.0 --max-grad-norm=0.7 \
	--lr=0.002 --phys-batch-size=256

# Evaluation of the full model
${PYTHON_CMD} evaluate.py --storage=${STORAGE} \
	--model-dir=${MODEL_DIR} --silo=Code15_test --subset-name=${SUBSET_NAME} \
	--mode full --num-splits=1
${PYTHON_CMD} evaluate.py --storage=${STORAGE} \
	--model-dir=${MODEL_DIR} --silo=Mimic_test --subset-name=${SUBSET_NAME} \
	--mode full --num-splits=1
${PYTHON_CMD} evaluate.py --storage=${STORAGE} \
	--model-dir=${MODEL_DIR} --silo=Hefei_compat --subset-name=${SUBSET_NAME} \
	--mode full --num-splits=1
cp -v ${STORAGE}/${MODEL_DIR}/${LOG_DIR}/evaluation.json ${RESULTS_DIR}/eval_full.json

# Feature extraction for fast finetuning
extractor="classif"
${PYTHON_CMD} feature_extract.py --storage=${STORAGE} \
	--model-dir=${MODEL_DIR} --silo=Code15 --subset-name=${SUBSET_NAME} --extractor=${extractor}
${PYTHON_CMD} feature_extract.py --storage=${STORAGE} \
	--model-dir=${MODEL_DIR} --silo=Mimic --subset-name=${SUBSET_NAME} --extractor=${extractor}
${PYTHON_CMD} feature_extract.py --storage=${STORAGE} \
	--model-dir=${MODEL_DIR} --silo=Hefei --subset-name=${SUBSET_NAME} --extractor=${extractor}

# Finetune the classifier
${PYTHON_CMD} finetune.py --storage=${STORAGE} \
	--model-dir=${MODEL_DIR} --silo=Code15  \
	--train-set=train --local-epoch=200
${PYTHON_CMD} finetune.py --storage=${STORAGE} \
	--model-dir=${MODEL_DIR} --silo=Mimic  \
	--train-set=train --local-epoch=200
${PYTHON_CMD} finetune.py --storage=${STORAGE} \
	--model-dir=${MODEL_DIR} --silo=Hefei  \
	--train-set=train --local-epoch=200

${PYTHON_CMD} finetune.py --storage=${STORAGE} \
	--model-dir=${MODEL_DIR} --silo=Code15  \
	--train-set=train_1K --local-epoch=200
${PYTHON_CMD} finetune.py --storage=${STORAGE} \
	--model-dir=${MODEL_DIR} --silo=Mimic  \
	--train-set=train_1K --local-epoch=200
${PYTHON_CMD} finetune.py --storage=${STORAGE} \
	--model-dir=${MODEL_DIR} --silo=Hefei  \
	--train-set=train_1K --local-epoch=200

# Test finetuned models
for test_silo in Code15 Mimic Hefei; do
	${PYTHON_CMD} ./evaluate.py --storage ${STORAGE} --model-dir=${MODEL_DIR} \
		--train-set=train --silo=${test_silo}
	${PYTHON_CMD} ./evaluate.py --storage ${STORAGE} --model-dir=${MODEL_DIR} \
		--train-set=train_1K --silo=${test_silo}
done
cp -v ${STORAGE}/${MODEL_DIR}/${LOG_DIR}/evaluation.json ${RESULTS_DIR}/eval_finetuned.json

