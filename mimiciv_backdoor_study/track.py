from pathlib import Path
import json
import shutil
import time

class FileTracker:
    """
    Minimal file-based tracker.

    Usage:
      tracker = FileTracker(run_dir)
      tracker.start_run(config=vars(args), metadata={...})
      tracker.log_metrics({"train_loss": x, "val_auroc": y}, step=epoch)
      tracker.log_artifact("path/to/model.pt", "model.pt")
      tracker.finish_run()
    """
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.started = False
        self.metrics_path = self.artifacts_dir / "metrics.json"
        self.config_path = self.artifacts_dir / "config.json"
        self.metadata_path = self.artifacts_dir / "metadata.json"

    def start_run(self, config: dict = None, metadata: dict = None):
        if config is not None:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)
        if metadata is not None:
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        # initialize metrics file as dict keyed by step
        if not self.metrics_path.exists():
            with open(self.metrics_path, "w") as f:
                json.dump({}, f, indent=2)
        self.started = True
        self._write_marker("run_started")

    def log_metrics(self, metrics: dict, step=None):
        if not self.started:
            # allow logging even if start_run wasn't called explicitly
            self.start_run()
        try:
            with open(self.metrics_path, "r") as f:
                cur = json.load(f)
        except Exception:
            cur = {}
        key = f"step_{step}" if step is not None else f"ts_{int(time.time())}"
        # ensure values are JSON serializable (caller should pass primitives)
        cur[key] = metrics
        with open(self.metrics_path, "w") as f:
            json.dump(cur, f, indent=2)

    def log_artifact(self, src_path: str, dest_name: str = None):
        src = Path(src_path)
        if not src.exists():
            return False
        if dest_name is None:
            dest_name = src.name
        dest = self.artifacts_dir / dest_name
        try:
            shutil.copy2(str(src), str(dest))
            return True
        except Exception:
            try:
                # fallback: write bytes
                data = src.read_bytes()
                dest.write_bytes(data)
                return True
            except Exception:
                return False

    def finish_run(self):
        self._write_marker("run_finished")

    def _write_marker(self, name: str):
        try:
            with open(self.artifacts_dir / f".{name}", "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass
