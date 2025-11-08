# mimiciv_backdoor_study

Reproducible research scaffold for the thesis "Backdoor Vulnerabilities in Deep Learning Models for Clinical Prediction: A Case Study on MIMIC-IV-Ext-CEKG".

Quickstart
1. Create a virtualenv at .venv (recommended) and install dependencies:
   - python3 -m venv .venv
   - source .venv/bin/activate    # or: . .venv/bin/activate
   - python3 -m pip install --upgrade pip
   - pip install -r requirements.txt
2. Build dev subset (synthetic placeholder for initial runs):
   - . .venv/bin/activate && python scripts/02_sample_dev.py
3. Run smoke test (end-to-end sanity check):
   - . .venv/bin/activate && bash tests/smoke_test.sh
4. Train (example):
   - . .venv/bin/activate && python train.py model=mlp trigger=none poison_rate=0.0
5. Evaluate:
   - . .venv/bin/activate && python eval.py run_path=runs/mlp/none/0.0/
6. Detect:
   - . .venv/bin/activate && python detect.py run_path=runs/mlp/rare_value/0.01/

Repository layout
```
mimiciv_backdoor_study/
├── data/
│   ├── raw/
│   ├── parquet/
│   ├── dev/
│   └── splits/
├── scripts/
│   ├── 00_to_parquet.py
│   ├── 01_build_cohort.sql
│   └── 02_sample_dev.py
├── models/
│   └── mlp.py
├── data_utils/
│   ├── dataset.py
│   └── triggers.py
├── configs/
│   ├── base.yaml
│   └── model/
│       └── mlp.yaml
├── train.py
├── eval.py
├── detect.py
├── requirements.txt
└── README.md
```

Notes
- This scaffold includes minimal synthetic-data implementations so training runs end-to-end on a small dev subset. Hooks and comments indicate where to integrate real MIMIC-IV-Ext-CEKG processing (PhysioNet credentials required).
- Use Hydra config overrides via CLI, e.g. `python train.py model=mlp trigger=rare_value poison_rate=0.01`.
