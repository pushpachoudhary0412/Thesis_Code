# progress.md

Snapshot of progress and actions (up to commit a449250)

Date: 2025-11-08
Branch: fix/bench-plot-pylance

Summary
- Fixed an editor/linter warning and made small CI/workflow improvements. All changes are committed on branch fix/bench-plot-pylance.

Timeline of important changes
- 2025-11-08 — a449250
  - Added minimal pyproject.toml so CI can run `pip install -e .` (resolves CI error when workflow installs repo root).
- 2025-11-08 — fc90b3f
  - Updated memory-bank/activeContext.md with a concise activity summary.
- 2025-11-08 — 56a543d
  - Silenced Pylance unresolved-import warning in mimiciv_backdoor_study/data_utils/dataset.py by adding:
    - import numpy as np  # type: ignore
  - Committed and pushed dataset change.

Other changes added to repo in this work
- Added memory-bank core files:
  - memory-bank/projectbrief.md
  - memory-bank/productContext.md
  - memory-bank/systemPatterns.md
  - memory-bank/techContext.md
  - memory-bank/activeContext.md (updated)
  - memory-bank/progress.md (this file)
- Minor CI workflow edit (previous commit 88cdfe6) to ensure repo is installed during CI to avoid import issues.

Local verification and status
- Ran a smoke test locally: python mimiciv_backdoor_study/data_utils/dataset.py
  - Observed output:
    - Dataset length 14000
    - Sample x shape torch.Size([30])
    - Triggered sample x shape torch.Size([30])
  - There were non-fatal pandas warnings about numexpr and bottleneck versions on the local environment (recommend upgrading those packages if desired).
- CI note:
  - Prior CI run failed because the workflow attempted to install the repo root but the repository had no packaging metadata. This is resolved by adding pyproject.toml. Recommend re-running CI to confirm.

Next steps / recommendations
- Re-run CI to confirm smoke workflow passes.
- Optional: pin minimal packaging metadata fields (authors/contact) or refine pyproject.toml metadata.
- Optional: update CI to install requirements directly instead of installing the repo root:
  - python -m pip install -r mimiciv_backdoor_study/requirements.txt
- Continue updating memory-bank files after any subsequent PRs/commits so the Memory Bank remains the canonical project state.

This file is intended to be an evolving, factual timeline of project progress. Update it each time the repository state changes significantly.
