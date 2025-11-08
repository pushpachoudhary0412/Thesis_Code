# Data README

Purpose
This document explains how to use a local copy of the MIMIC-IV dataset archive for local experiments, how to regenerate the Parquet dataset used by the study, and how to avoid committing large data files to git.

Local ZIP placement (do NOT commit)
- Place the downloaded MIMIC-IV ZIP file on your machine at:
  mimiciv_backdoor_study/data/raw/mimic-iv-3.1.zip
- Do not add or commit the ZIP or extracted raw files to the repository.

Unzip locally
Example (from repo root):
1. Create target raw directory (if necessary):
   mkdir -p mimiciv_backdoor_study/data/raw
2. Unzip the archive into a folder named mimic-iv-3.1:
   unzip /path/to/mimic-iv-3.1.zip -d mimiciv_backdoor_study/data/raw

Generate Parquet & splits (using the project's script)
- The repository contains a helper script that converts raw CSV/GZ files into Parquet and writes a splits JSON:
  python mimiciv_backdoor_study/scripts/00_to_parquet.py
- Example usage (from repo root):
  unzip mimiciv_backdoor_study/data/raw/mimic-iv-3.1.zip -d mimiciv_backdoor_study/data/raw
  python mimiciv_backdoor_study/scripts/00_to_parquet.py
- Output (by default) will be written to:
  - mimiciv_backdoor_study/data/main.parquet
  - mimiciv_backdoor_study/data/splits_main.json

Quick local smoke (after generating Parquet)
- You can run the repo's small smoke tests or dataset/demo scripts that expect the Parquet file:
  python mimiciv_backdoor_study/data_utils/dataset.py
  scripts/smoke_test.sh

Safety / Git
- Recommended .gitignore entries to avoid accidentally committing large files:
  # ignore raw datasets and archives
  mimiciv_backdoor_study/data/raw/
  *.zip
  *.tar.gz

Notes
- This README and any helper scripts are safe to commit. The raw dataset must remain local and gitignored.
- If you want, the repository can include a small helper script (scripts/fetch_data.sh) with documented commands for unzipping / preparing the data locally. I can add that and the .gitignore updates and commit them on your confirmation.
