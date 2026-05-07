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

# Build downloader.py CLI args from config.
ARGS=$(python - "$CONFIG" <<'PY'
import sys, shlex, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
s, p, d = cfg["shared"], cfg["paths"], cfg["download"]

pairs = [
    ("gps_start", d["gps_start"]),
    ("gps_end", d["gps_end"]),
    ("sample_rate", s["sample_rate"]),
    ("window_len_s", d["window_len_s"]),
    ("mode", d["mode"]),
    ("n_segments", d["n_segments"]),
    ("output_dir", p["noise_dir"]),
    ("band_low", s["band_low"]),
    ("band_high", s["band_high"]),
    ("bandpass_order", s["bandpass_order"]),
    ("psd_seglen_s", d["psd_seglen_s"]),
    ("target_plot_fs", d["target_plot_fs"]),
    ("event_pad_s", d["event_pad_s"]),
    ("glitch_sigma", d["glitch_sigma"]),
    ("glitch_max_frac", d["glitch_max_frac"]),
    ("max_std_ratio", d["max_std_ratio"]),
]

for key in ["amp_thresh", "rms_thresh", "max_raw_std", "min_raw_std"]:
    if d.get(key) is not None:
        pairs.append((key, d[key]))

args = []
for key, value in pairs:
    args += [f"--{key}", str(value)]

for key, flag in [
    ("require_full_window", "--require_full_window"),
    ("plot_timeline", "--plot_timeline"),
    ("plot_timeseries", "--plot_timeseries"),
    ("plot_psd", "--plot_psd"),
]:
    if d.get(key, False):
        args.append(flag)

if s.get("noise_is_whitened", False):
    args.append("--whiten")

print(" ".join(shlex.quote(a) for a in args))
PY
)

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
gpus_per_node: ${SLURM_GPUS_PER_NODE:-unknown}
partition: ${SLURM_JOB_PARTITION:-unknown}
account: ${SLURM_JOB_ACCOUNT:-unknown}
python: $(which python)
config_source: ${CONFIG}
command: python downloader.py ${ARGS}
EOF

echo "Running:"
echo "python downloader.py $ARGS"

srun python -u downloader.py $ARGS
