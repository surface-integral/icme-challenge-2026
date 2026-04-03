#!/usr/bin/env bash
# scripts/2_preprocess.sh
#
# Runs the NTU ATTM vocal-separation preprocessing pipeline.
# Clones the official NTU preprocessing repo and runs it on the dataset.
#
# Usage:
#   bash scripts/2_preprocess.sh [CREATE_SUBSET] [INPUT_DIR] [OUTPUT_DIR] [MAX_JOBS]
#
# Arguments:
#   CREATE_SUBSET — Whether to create a 30s subset of each audio file (default: true)
#   INPUT_DIR  — Path to the downloaded MTG-Jamendo audio  (default: ../mtg_jamendo)
#   OUTPUT_DIR — Where to save separated audio             (default: ../mtg_jamendo_separated)
#   MAX_JOBS   — Parallel folder jobs (each uses ~4GB VRAM, default: 4)
#
# Notes:
#   • Processing 1 folder (~500 files) takes ~15 hours on a single GPU.
#   • With MAX_JOBS=4, all 100 folders take roughly 15 days total wall-clock.
#   • For resource-constrained setups, create a 30s subset first (see below).
#   • The NTU script only keeps the non-vocal (instrumental) stems.

set -euo pipefail

CREATE_SUBSET="${1:-true}"
INPUT_DIR="${2:-../mtg_jamendo}"
OUTPUT_DIR="${3:-../mtg_jamendo_separated}"
MAX_JOBS="${4:-4}"

echo "============================================================"
echo " ATTM Grand Challenge — Step 2: Vocal Separation"
echo "============================================================"
echo " Creating subset: $CREATE_SUBSET"
echo " Input  : $INPUT_DIR"
echo " Output : $OUTPUT_DIR"
echo " Parallel jobs : $MAX_JOBS  (~$((MAX_JOBS * 4)) GB VRAM required)"
echo ""

# ── 1. Clone NTU preprocessing repo ─────────────────────────────────────
if [ ! -d "ICME26-ATTM-GC-Preprocessing" ]; then
    echo "[1/4] Cloning NTU preprocessing repository..."
    git clone https://github.com/ntu-musicailab/ICME26-ATTM-GC-Preprocessing.git
else
    echo "[1/4] NTU preprocessing repo already exists."
fi

cd ICME26-ATTM-GC-Preprocessing

# ── 2. Install melband-roformer ─────────────────────────────────────────
echo ""
echo "[2/4] Installing melband-roformer-infer..."
pip install melband-roformer-infer pydub tqdm --quiet

# ── 3. Download model weights ────────────────────────────────────────────
echo ""
echo "[3/4] Downloading Mel-Band Roformer vocal separation weights..."
mkdir -p ./models
melband-roformer-download --model melband-roformer-kim-vocals --output-dir ./models

if CREATE_SUBSET; then
    echo ""
    echo "Creating 30s subsets of each audio file for faster processing..."
    mkdir -p ${INPUT_DIR}_subset_30s
    python create_subset.py ${INPUT_DIR} ${INPUT_DIR}_subset_30s --num_seconds 30 --num_workers ${MAX_JOBS}
    INPUT_DIR="${INPUT_DIR}_subset_30s"
else
    echo ""
    echo "Skipping subset creation. Processing full audio files (this may take a long time)..."
fi

# ── 4. Run parallel separation ───────────────────────────────────────────
echo ""
echo "[4/4] Starting parallel vocal separation..."
echo "      Logs will be written to ./logs/"
echo ""

# Patch the config variables in process_parallel.sh before running
sed \
    -e "s|MAX_JOBS=10|MAX_JOBS=${MAX_JOBS}|" \
    -e "s|DATASET_ROOT=\"../mtg_jamendo\"|DATASET_ROOT=\"${INPUT_DIR}\"|" \
    -e "s|OUTPUT_ROOT=\"../mtg_jamendo_separated\"|OUTPUT_ROOT=\"${OUTPUT_DIR}\"|" \
    process_parallel.sh > process_parallel_configured.sh

bash process_parallel_configured.sh

cd ..

echo ""
echo "============================================================"
echo " Preprocessing complete!"
echo " Separated audio is at: $OUTPUT_DIR"
echo ""
echo " Next step: python scripts/3_encode_latents.py \\"
echo "              --audio_dir $OUTPUT_DIR \\"
echo "              --output_dir ../mtg_latents"
echo "============================================================"
