"""
sweepmanager.py
===============
Parameter-sweep orchestrator for luminis-mc simulations.

:class:`SweepManager` coordinates a series of :class:`~luminis_mc.manager.Experiment`
runs (a *sweep*) under a shared root directory.  Each run gets its own
timestamped sub-folder and is tracked in a newline-delimited JSON manifest
(``sweep_manifest.jsonl``) that records start time, finish time, runtime,
and status (``"ok"`` or ``"failed"``).

Directory layout produced by ``SweepManager``::

    <base_dir>/<YYYY-MM-DD_HH-MM-SS>_<sweep_name>/
    ├── sweep_manifest.jsonl   # one JSON record per run (started + finished)
    ├── script_snapshot.py     # copy of the master sweep script
    ├── README.md              # optional free-text notes
    └── runs/
        ├── 0000_<run_name>/   # individual Experiment directory
        │   └── results.h5
        ├── 0001_<run_name>/
        ...

Usage example::

    sweep = SweepManager("mfp_sweep", base_dir="/data/results")
    sweep.snapshot_master_script(__file__)

    for i, mfp in enumerate([0.5, 1.0, 2.0]):
        def run_fn(exp, mfp=mfp):
            medium = MieMedium(..., mfp=mfp, ...)
            config = SimConfig(...)
            run_simulation_parallel(config)
            exp.save_sensor(sensor, "far_field_cbs")
            exp.log_params(mean_free_path=mfp)

        sweep.run(run_id=i, run_name=f"mfp_{mfp}", run_fn=run_fn)
"""

import json
import time
from pathlib import Path
from datetime import datetime

from .manager import Experiment


# ══════════════════════════════════════════════════════════════════════════════
#  SweepManager
# ══════════════════════════════════════════════════════════════════════════════

class SweepManager:
    """
    Orchestrates a parameter sweep as a collection of :class:`Experiment` runs.

    Each call to :py:meth:`run` creates a self-contained sub-experiment,
    appends two records to the manifest (``"started"`` then ``"ok"`` /
    ``"failed"``), and re-raises any exception after marking the run as
    failed so the caller can decide whether to abort the sweep.
    """

    def __init__(self, sweep_name: str, base_dir: str):
        """
        Create the sweep root directory and initialise the manifest file.

        Parameters
        ----------
        sweep_name:
            Human-readable label appended to the timestamped directory name.
        base_dir:
            Parent directory under which the sweep folder is created.
        """
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.root     = Path(base_dir) / f"{ts}_{sweep_name}"
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_jsonl = self.root / "sweep_manifest.jsonl"
        self.readme         = self.root / "README.md"

    # ── Directory-level utilities ──────────────────────────────────────────────

    def log_readme(self, text: str) -> None:
        """Write *text* to the sweep-level ``README.md``."""
        self.readme.write_text(text, encoding="utf-8")

    def snapshot_master_script(self, file_path: str) -> None:
        """
        Copy the master sweep script to the sweep root directory once.

        This preserves the exact parameter-sweep logic used to generate
        all runs in this sweep.

        Parameters
        ----------
        file_path:
            Absolute path of the script to copy (typically ``__file__``).
        """
        dst = self.root / "script_snapshot.py"
        dst.write_text(
            Path(file_path).read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )

    # ── Manifest ───────────────────────────────────────────────────────────────

    def _append_manifest(self, record: dict) -> None:
        """Append a single JSON record (one line) to ``sweep_manifest.jsonl``."""
        with open(self.manifest_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ── Run execution ──────────────────────────────────────────────────────────

    def run(self, run_id: int, run_name: str, run_fn) -> None:
        """
        Execute a single sweep run inside its own :class:`Experiment`.

        The caller supplies *run_fn*, a callable that receives an open
        :class:`Experiment` instance and is responsible for:

        1. Building the medium, laser, sensors, and :class:`SimConfig`.
        2. Calling :func:`run_simulation_parallel` (or the single-threaded
           variant).
        3. Persisting results via ``exp.save_sensor(...)`` and optionally
           ``exp.save_derived(...)`` and ``exp.log_params(...)``.

        The experiment is closed automatically after *run_fn* returns.
        Two manifest records are written: one at start (``"started"``) and
        one at finish (``"ok"`` or ``"failed"``).

        Parameters
        ----------
        run_id:
            Zero-based integer index used to sort runs in the manifest and
            to name the sub-directory (zero-padded to four digits).
        run_name:
            Human-readable label for this run, appended after the run index
            in the directory name.
        run_fn:
            Callable with signature ``run_fn(exp: Experiment) -> None``.

        Raises
        ------
        RuntimeError:
            Wraps any exception raised inside *run_fn*, after marking the
            run as ``"failed"`` in the manifest.
        """
        run_folder = self.runs_dir / f"{run_id:04d}_{run_name}"
        run_folder.mkdir(parents=True, exist_ok=True)

        t0     = time.time()
        status = "ok"
        err    = None

        record = {
            "run_id":     run_id,
            "run_name":   run_name,
            "path":       str(run_folder),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._append_manifest({**record, "status": "started"})

        try:
            # Each run gets its own Experiment rooted inside runs_dir.
            exp = Experiment(name=run_folder.name, base_dir=str(self.runs_dir))
            run_fn(exp)
            exp.close()
        except Exception as e:
            status = "failed"
            err    = repr(e)

        dt = time.time() - t0
        self._append_manifest({
            **record,
            "status":      status,
            "error":       err,
            "runtime_s":   dt,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        })

        if status != "ok":
            raise RuntimeError(f"Run {run_id} ('{run_name}') failed: {err}")
