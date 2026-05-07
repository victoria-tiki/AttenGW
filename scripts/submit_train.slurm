#!/bin/bash
#SBATCH --job-name=AttenGW
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# Submit from the repository root:
#   sbatch scripts/submit.slurm

CONFIG="configs/example.yaml"
JOB_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"

python -c "import yaml" 2>/dev/null || pip install --user pyyaml

# Create one checkpoint subfolder per job.
BASE_CHECKPOINT_DIR=$(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg["paths"]["checkpoint_dir"])
PY
)

RUN_ID="train_job_${JOB_ID}"
CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR%/}/${RUN_ID}"
mkdir -p "$CHECKPOINT_DIR"

# Build train.py CLI args from config + current SLURM allocation.
ARGS=$(python - "$CONFIG" "$CHECKPOINT_DIR" "$NUM_NODES" <<'PY'
import sys, shlex, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
checkpoint_dir, num_nodes = sys.argv[2], sys.argv[3]
s, p, t = cfg["shared"], cfg["paths"], cfg["training"]

pairs = [
    ("batch_size", t["batch_size"]),
    ("data_dir", p["data_dir"]),
    ("checkpoint_dir", checkpoint_dir),
    ("noise_dir", p["noise_dir"]),
    ("num_workers", t["num_workers"]),
    ("num_nodes", num_nodes),
    ("lr_init", t["lr_init"]),
    ("dim", t["dim"]),
    ("segment_length", t["segment_length"]),
    ("edge_buffer", t["edge_buffer"]),
    ("sample_rate", s["sample_rate"]),
    ("band_low", s["band_low"]),
    ("band_high", s["band_high"]),
    ("bandpass_order", s["bandpass_order"]),
    ("psd_floor", s["psd_floor"]),
    ("psd_outband", s["psd_outband"]),
    ("noise_prob", t["noise_prob"]),
    ("p_higher_init", t["p_higher_init"]),
    ("p_higher_fin", t["p_higher_fin"]),
    ("train_file", t["train_file"]),
    ("val_file", t["val_file"]),
]

args = []
for key, value in pairs:
    args += [f"--{key}", str(value)]

if s.get("noise_is_whitened", False):
    args.append("--noise_is_whitened")

print(" ".join(shlex.quote(a) for a in args))
PY
)

# Save provenance in the same folder as checkpoints.
cp "$CONFIG" "$CHECKPOINT_DIR/config.yaml"
cat > "$CHECKPOINT_DIR/runinfo.txt" <<EOF
run_id: ${RUN_ID}
job_id: ${SLURM_JOB_ID:-unknown}
job_name: ${SLURM_JOB_NAME:-unknown}
submit_dir: ${SLURM_SUBMIT_DIR:-unknown}
work_dir: $(pwd)
hostname: $(hostname)
date: $(date)
nodes: ${SLURM_JOB_NUM_NODES:-unknown}
node_list: ${SLURM_JOB_NODELIST:-unknown}
cpus_per_task: ${SLURM_CPUS_PER_TASK:-unknown}
gpus_per_node: ${SLURM_GPUS_PER_NODE:-unknown}
partition: ${SLURM_JOB_PARTITION:-unknown}
account: ${SLURM_JOB_ACCOUNT:-unknown}
python: $(which python)
config_source: ${CONFIG}
command: python train.py ${ARGS}
EOF

echo "Running:"
echo "python train.py $ARGS"

srun python -u train.py $ARGS
