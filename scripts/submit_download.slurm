#!/bin/bash
#SBATCH --job-name=GWDownload
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# Submit from the repository root:
#   sbatch scripts/submit_download.slurm

CONFIG="configs/example.yaml"
JOB_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"

python -c "import yaml" 2>/dev/null || pip install --user pyyaml

# Read downloader output directory from config.
NOISE_DIR=$(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["paths"]["noise_dir"])
PY
)

mkdir -p "$NOISE_DIR"

# Save provenance next to generated noise files.
cp "$CONFIG" "$NOISE_DIR/config_download_job_${JOB_ID}.yaml"

cat > "$NOISE_DIR/runinfo_download_job_${JOB_ID}.txt" <<EOF
run_id: download_job_${JOB_ID}
job_id: ${SLURM_JOB_ID:-unknown}
job_name: ${SLURM_JOB_NAME:-unknown}
submit_dir: ${SLURM_SUBMIT_DIR:-unknown}
work_dir: $(pwd)
hostname: $(hostname)
date: $(date)
nodes: ${SLURM_JOB_NUM_NODES:-unknown}
node_list: ${SLURM_JOB_NODELIST:-unknown}
cpus_per_task: ${SLURM_CPUS_PER_TASK:-unknown}
partition: ${SLURM_JOB_PARTITION:-unknown}
account: ${SLURM_JOB_ACCOUNT:-unknown}
python: $(which python)
config_source: ${CONFIG}
command: python downloader.py --config ${CONFIG}
EOF

echo "Running:"
echo "python downloader.py --config $CONFIG"

srun python -u downloader.py --config "$CONFIG"
