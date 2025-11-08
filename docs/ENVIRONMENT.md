# Reproducible environment

This project requires a Python environment with binary deps (notably pyarrow). Recommended approach is to use conda (conda-forge) to avoid fragile wheel builds on macOS.

Quick steps (recommended)
1. Create the conda environment from the provided lockfile (preferred):
   conda env create -f environment-locked.yml

2. Alternatively create from the human-friendly spec then install pip deps:
   conda env create -f environment.yml
   conda run -n mimiciv_env pip install -r mimiciv_backdoor_study/requirements.txt

3. Run project scripts using the environment's python and set PYTHONPATH:
   PYTHONPATH=$(pwd) conda run -n mimiciv_env python mimiciv_backdoor_study/scripts/02_sample_dev.py
   PYTHONPATH=$(pwd) conda run -n mimiciv_env python mimiciv_backdoor_study/train.py --model mlp --epochs 2
   PYTHONPATH=$(pwd) conda run -n mimiciv_env python mimiciv_backdoor_study/eval.py --run_dir mimiciv_backdoor_study/runs/...
   PYTHONPATH=$(pwd) conda run -n mimiciv_env python mimiciv_backdoor_study/detect.py --run_dir mimiciv_backdoor_study/runs/... --method activation_clustering

Notes about the devcontainer
- The included devcontainer currently installs pip requirements at build-time (Dockerfile). That installs pure-Python deps but may fail for binary packages like pyarrow on certain hosts.
- Options:
  - Open the repository in the devcontainer and manually create the conda env inside the container:
    conda env create -f /workspace/environment.yml
    conda run -n mimiciv_env pip install -r /workspace/mimiciv_backdoor_study/requirements.txt
  - Or run the recommended conda env on the host (local dev) and use VSCode remote interpreter set to that env.

CI notes
- GitHub Actions workflow (/.github/workflows/smoke.yml) installs pip deps and runs smoke tests and pytest on ubuntu-latest. This uses pip because Linux wheels avoid many macOS build issues.
- For macOS local development, prefer conda-forge pyarrow as shown above.

Threading / OpenMP notes (macOS and CI)
- Some binary numeric packages (scikit-learn, numpy, scikit-learn's KMeans) use OpenMP and can crash when multiple OpenMP runtimes are loaded (e.g., Intel libiomp + LLVM libomp). This can cause intermittent segmentation faults on macOS and other platforms.
- Workaround (recommended): limit BLAS / OpenMP threads when running tests, benchmarks, or other CPU-bound code. Example (POSIX shells):
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1

- Persist this in CI: add these env vars to your GitHub Actions job (env: OMP_NUM_THREADS: "1", MKL_NUM_THREADS: "1", OPENBLAS_NUM_THREADS: "1") so runners do not mix OpenMP runtimes during tests.
- Prefer installing binary packages from conda-forge for macOS (pyarrow, numpy, scikit-learn) to reduce likelihood of ABI/runtime mismatches.

- Alternative mitigations:
  - Use a single OpenMP provider in your environment (avoid mixing packages built against different providers).
  - Run heavy numerical workloads inside a dedicated process with controlled environment variables.

If you'd like, I can:
- Update the devcontainer Dockerfile to install micromamba and create the conda env automatically at build time (bakes in binary deps).
- Replace the devcontainer postCreateCommand to guide users to create the conda env inside the container.
- Further pin pip versions in requirements.txt or produce a pip freeze fragment.

If you'd like, I can:
- Update the devcontainer Dockerfile to install micromamba and create the conda env automatically at build time (bakes in binary deps).
- Replace the devcontainer postCreateCommand to guide users to create the conda env inside the container.
- Further pin pip versions in requirements.txt or produce a pip freeze fragment.

Which of these should I do next?
