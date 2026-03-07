#!/bin/bash
set -eo pipefail

mkdir -p jobs/output

# -----------------------
# User-editable settings
# -----------------------
CONFIG="configs/run.yaml"
TASK_NAME="sir"
MODEL="x0x1"          # x0x1 | x1 | velocity
INIT_DIST="gaussian"  # gaussian | theta_prior
N_TRAIN="100000"
# -----------------------

JOB_NAME="run_${MODEL}_${INIT_DIST}_${TASK_NAME}_n${N_TRAIN}"
mkdir -p "jobs/output/${JOB_NAME}"

echo "Submitting single run: task=${TASK_NAME} model=${MODEL} init=${INIT_DIST} n_train=${N_TRAIN}"

sbatch \
  --job-name="${JOB_NAME}" \
  --export=CONFIG="${CONFIG}",TASK_NAME="${TASK_NAME}",MODEL="${MODEL}",INIT_DIST="${INIT_DIST}",N_TRAIN="${N_TRAIN}" \
  jobs/train_eval.job