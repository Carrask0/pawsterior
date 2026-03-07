# jobs/launch_sweep.sh
#!/bin/bash
set -eo pipefail

mkdir -p jobs/output

# -----------------------
# User-editable settings
# -----------------------
# SWEEP_CONFIG="configs/sweeps/sbibm_sweep.yaml"
SWEEP_CONFIG="configs/sweeps/sgm_sweep.yaml" 


BASE="$(basename "$SWEEP_CONFIG" .yaml)"
JOB_NAME="sweep_${BASE}"

mkdir -p "jobs/output/${JOB_NAME}"

echo "Submitting streaming sweep array with config: ${SWEEP_CONFIG}"
sbatch \
  --job-name="${JOB_NAME}" \
  --export=SWEEP_CONFIG="${SWEEP_CONFIG}" \
  jobs/sweep_array.job