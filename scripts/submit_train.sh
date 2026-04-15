#!/usr/bin/env bash
# Slurm batch script — ICU delirium T-PatchGNN training on Oscar HPC (Brown)
#
# Submit:  sbatch scripts/submit_train.sh
# Dry-run: sbatch --test-only scripts/submit_train.sh
#
# Verify GPU partition with:  sinfo -s | grep gpu
#
#SBATCH --job-name=delirium_train
#SBATCH --partition=gpu           # <-- adjust if needed (e.g. gpu-he, 3090-gcondo)
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4         # 2 DataLoader workers × 2 loaders (train + val)
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=syang195@brown.edu

set -euo pipefail

PROJECT=/oscar/home/syang195/1595-final
cd "$PROJECT"

# ── Activate virtual environment ───────────────────────────────────────────
source "$PROJECT/.venv/bin/activate"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# ── GPU diagnostics ────────────────────────────────────────────────────────
echo "SLURM_JOB_ID:   $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
"

# ── Create output directories ──────────────────────────────────────────────
mkdir -p "$PROJECT/checkpoints"
mkdir -p "$PROJECT/results"
mkdir -p "$PROJECT/logs"

# ── Train ──────────────────────────────────────────────────────────────────
python -m src.train \
    --cohort       "$PROJECT/cohort.csv" \
    --features     "$PROJECT/features_hourly.csv" \
    --output-dir   "$PROJECT/checkpoints" \
    --max-hours    24 \
    --epochs       50 \
    --batch-size   32 \
    --hid-dim      32 \
    --n-layer      2 \
    --nhead        4 \
    --tf-layer     2 \
    --node-dim     10 \
    --dropout      0.1 \
    --lr           1e-3 \
    --grad-clip    1.0 \
    --lr-factor    0.5 \
    --lr-patience  5 \
    --lr-min       1e-5 \
    --patience     10 \
    --min-delta    1e-4 \
    --bootstrap-iters 200 \
    --history-csv  "$PROJECT/results/training_history.csv" \
    --predictions-csv "$PROJECT/results/test_predictions.csv"

echo "Training complete."
echo "Checkpoint  : $PROJECT/checkpoints/best_model.pt"
echo "History     : $PROJECT/results/training_history.csv"
echo "Predictions : $PROJECT/results/test_predictions.csv"

# ── Generate visualisation plots ──────────────────────────────────────────
python -c "
from src.viz import make_all_plots
make_all_plots(
    'results/test_predictions.csv',
    'results/training_history.csv',
    output_dir='results',
    show=False,
)
"
echo "Plots saved to $PROJECT/results/"
