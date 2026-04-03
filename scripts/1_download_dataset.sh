#!/usr/bin/env bash
# scripts/1_download_dataset.sh
#
# Downloads the MTG-Jamendo raw_30s audio subset.
# Uses the fast MTG mirror in Finland (default).
#
# Usage:
#   bash scripts/1_download_dataset.sh [OUTPUT_DIR]
#
# Default output: ../mtg_jamendo  (sibling of this project)

set -euo pipefail

OUTPUT_DIR="${1:-../mtg_jamendo}"

echo "============================================================"
echo " ATTM Grand Challenge — Step 1: Dataset Download"
echo "============================================================"
echo " Output directory : $OUTPUT_DIR"
echo ""

# ── 1. Clone MTG-Jamendo repo ────────────────────────────────────────────
if [ ! -d "mtg-jamendo-dataset" ]; then
    echo "[1/3] Cloning MTG-Jamendo repository..."
    git clone https://github.com/MTG/mtg-jamendo-dataset.git
else
    echo "[1/3] MTG-Jamendo repo already exists. Pulling latest..."
    git -C mtg-jamendo-dataset pull
fi

# ── 2. Install download dependencies ────────────────────────────────────
echo ""
echo "[2/3] Installing MTG-Jamendo download requirements..."
pip install -r mtg-jamendo-dataset/scripts/requirements.txt --quiet

# ── 3. Download raw_30s audio ────────────────────────────────────────────
echo ""
echo "[3/3] Downloading raw_30s audio subset (~120 GB compressed)..."
echo "      This will unpack tar archives and remove them to save space."
echo "      Estimated time: 1–6 hours depending on connection speed."
echo ""

mkdir -p "$OUTPUT_DIR"

python3 mtg-jamendo-dataset/scripts/download/download.py \
    --dataset raw_30s \
    --type audio \
    --from mtg-fast \
    --unpack \
    --remove \
    "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo " Download complete!"
echo " Audio files are at: $OUTPUT_DIR"
echo " Expected structure:"
echo "   $OUTPUT_DIR/00/*.mp3"
echo "   $OUTPUT_DIR/01/*.mp3"
echo "   ..."
echo "   $OUTPUT_DIR/99/*.mp3"
echo ""
echo " Next step: bash scripts/2_preprocess.sh $OUTPUT_DIR"
echo "============================================================"
