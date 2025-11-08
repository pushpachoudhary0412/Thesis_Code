#!/usr/bin/env bash
# scripts/fetch_data.sh
# Helper for local dataset preparation (DO NOT COMMIT DATA)
#
# Usage (from repo root):
#   chmod +x scripts/fetch_data.sh
#   ./scripts/fetch_data.sh /path/to/mimic-iv-3.1.zip
#
# This script will:
#  - create the local raw data folder
#  - unzip the provided archive into mimiciv_backdoor_study/data/raw/mimic-iv-3.1
#  - run the project's converter script to produce main.parquet and splits_main.json
#
# NOTE: Do not add or commit the raw archive or extracted files to git.

set -euo pipefail

ZIP_PATH="${1:-}"
RAW_DIR="mimiciv_backdoor_study/data/raw"
EXTRACT_DIR="$RAW_DIR/mimic-iv-3.1"

if [ -z "$ZIP_PATH" ]; then
  echo "Error: path to mimic-iv-3.1.zip is required."
  echo "Usage: ./scripts/fetch_data.sh /path/to/mimic-iv-3.1.zip"
  exit 2
fi

if [ ! -f "$ZIP_PATH" ]; then
  echo "Error: ZIP file not found at: $ZIP_PATH"
  exit 3
fi

echo "Creating raw data directory: $RAW_DIR"
mkdir -p "$RAW_DIR"

echo "Unzipping archive into: $RAW_DIR"
unzip -q "$ZIP_PATH" -d "$RAW_DIR"

echo "Ensuring extracted directory exists at: $EXTRACT_DIR"
if [ ! -d "$EXTRACT_DIR" ]; then
  echo "Warning: expected extracted folder at $EXTRACT_DIR not found."
  echo "List files in $RAW_DIR:"
  ls -la "$RAW_DIR"
  echo "Proceeding, but you may need to adjust the extraction target."
fi

echo "Running parquet conversion script"
python mimiciv_backdoor_study/scripts/00_to_parquet.py

echo "Done. Generated files (by default):"
echo "  - mimiciv_backdoor_study/data/main.parquet"
echo "  - mimiciv_backdoor_study/data/splits_main.json"
echo ""
echo "Reminder: Do NOT add or commit the raw ZIP or extracted files to git."
