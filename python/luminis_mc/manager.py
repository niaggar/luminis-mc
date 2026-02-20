"""
manager.py
==========
Experiment persistence layer for luminis-mc simulations.

Provides two public classes:

- **Experiment** — creates a timestamped output directory, writes all
  simulation parameters and sensor results to a single HDF5 file
  (``results.h5``), and copies the calling script as a snapshot for
  reproducibility.

- **ResultsLoader** — opens an existing ``results.h5`` and exposes
  convenient accessors for parameters, sensor data, and derived arrays.

HDF5 layout produced by ``Experiment``::

    results.h5
    ├── params/           # scalar attrs + array datasets of simulation params
    ├── sensors/
    │   └── <name>/
    │       ├── meta/     # sensor type, grid dims, filter settings (attrs)
    │       └── data/     # Stokes / field / record arrays (datasets)
    └── derived/          # post-processed arrays ready for plotting

Supported sensor types
----------------------
PhotonRecordSensor, PlanarFieldSensor, PlanarFluenceSensor, PlanarCBSSensor,
FarFieldFluenceSensor, FarFieldCBSSensor, StatisticsSensor.
To add a new sensor type extend the ``if/elif`` block in
:py:meth:`Experiment.save_sensor`.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import h5py


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _vec3_to_arr(v) -> np.ndarray:
    """Convert a Vec3 binding object to a (3,) float64 array."""
    return np.array([v.x, v.y, v.z], dtype=np.float64)


def _vec2_to_arr(v) -> np.ndarray:
    """Convert a Vec2 binding object to a (2,) float64 array."""
    return np.array([v.x, v.y], dtype=np.float64)


def _cvec2_to_arr(v) -> np.ndarray:
    """Convert a CVec2 binding object to a (2,) complex128 array."""
    return np.array([v.m, v.n], dtype=np.complex128)


def _is_scalar(x) -> bool:
    """Return True if *x* can be stored directly as an HDF5 attribute scalar."""
    return isinstance(x, (str, bool, int, float, np.integer, np.floating, np.bool_))


def _clean_scalar(x):
    """Convert numpy scalar types to their Python equivalents for HDF5 attrs."""
    return x.item() if isinstance(x, (np.integer, np.floating, np.bool_)) else x


def _as_array(x) -> Optional[np.ndarray]:
    """
    Coerce *x* to a numpy array, or return ``None`` if conversion is not
    possible.

    Handles the following input types:

    - :class:`numpy.ndarray` — returned as-is (unless dtype is object).
    - Binding objects with ``get_numpy()`` — calls that method.
    - ``list`` / ``tuple`` of numerics — passed to :func:`numpy.asarray`.
    - ``list`` / ``tuple`` of binding matrices — stacked into a 3-D array.
    """
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x if x.dtype != object else None
    if hasattr(x, "get_numpy"):
        try:
            a = np.asarray(x.get_numpy())
            return a if a.dtype != object else None
        except Exception:
            return None
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        # try a direct conversion first
        a = np.asarray(x)
        if a.dtype != object:
            return a
        # fall back to stacking individual matrices
        first = _as_array(x[0])
        if first is None:
            return None
        mats = []
        for y in x:
            ay = _as_array(y)
            if ay is None or ay.shape != first.shape or ay.dtype != first.dtype:
                return None
            mats.append(ay)
        return np.stack(mats, axis=0)
    return None


def _write_dataset(
    g: h5py.Group,
    name: str,
    arr,
    compression: str = "gzip",
    clevel: int = 4,
) -> h5py.Dataset:
    """
    Write (or overwrite) a dataset inside an HDF5 group.

    Compression is applied only when the array contains more than 256
    elements to avoid overhead for small arrays.

    Parameters
    ----------
    g:
        Parent HDF5 group.
    name:
        Dataset name within *g*.
    arr:
        Data to write; coerced with :func:`numpy.asarray`.
    compression:
        Compression filter name (default ``"gzip"``).
    clevel:
        Compression level 0–9 (default 4).

    Returns
    -------
    h5py.Dataset
        The newly created dataset.
    """
    if name in g:
        del g[name]
    arr = np.asarray(arr)
    use_comp = arr.size > 256
    return g.create_dataset(
        name,
        data=arr,
        compression=(compression if use_comp else None),
        compression_opts=(clevel if use_comp else None),
        shuffle=(True if use_comp else False),
    )


def _write_attr(g: h5py.Group, key: str, val: Any) -> None:
    """
    Write a single value as an HDF5 attribute on group *g*.

    Type dispatch:

    - ``None``                       → skipped.
    - Scalars (str, bool, int, …)    → stored directly.
    - Pybind11 enums (have ``.name``)→ stored as their string name.
    - Vec3                            → stored as (3,) float64 array.
    - Vec2                            → stored as (2,) float64 array.
    - CVec2                           → stored as (2,) complex128 array.
    - Anything else                   → ``str(val)`` fallback.
    """
    if val is None:
        return
    if _is_scalar(val):
        g.attrs[key] = _clean_scalar(val)
        return
    # pybind11 enums: store the enum name as a string
    if hasattr(val, "name"):
        try:
            g.attrs[key] = str(val.name)
            return
        except Exception:
            pass
    # Vec3 / Vec2 / CVec2
    if hasattr(val, "x") and hasattr(val, "y") and hasattr(val, "z"):
        g.attrs[key] = _vec3_to_arr(val)
        return
    if hasattr(val, "x") and hasattr(val, "y") and not hasattr(val, "z"):
        g.attrs[key] = _vec2_to_arr(val)
        return
    if hasattr(val, "m") and hasattr(val, "n"):
        g.attrs[key] = _cvec2_to_arr(val)
        return
    # generic fallback: store as string
    g.attrs[key] = str(val)


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment
# ══════════════════════════════════════════════════════════════════════════════

class Experiment:
    """
    Manages a single simulation experiment: directory creation, parameter
    logging, sensor persistence, and script snapshotting.

    The output directory is created at
    ``<base_dir>/<YYYY-MM-DD_HH-MM-SS>_<name>/`` and contains:

    - ``results.h5``          — all simulation data.
    - ``script_snapshot.py``  — copy of the script that ran the experiment.
    - ``README.md``           — optional free-text notes.

    Usage::

        exp = Experiment("my_run", base_dir="/data/results")
        exp.log_script(__file__)
        exp.log_params(n_photons=1_000_000, mu_s=1.0)
        run_simulation_parallel(config)
        exp.save_sensor(sensor, "far_field_cbs")
        exp.close()

    Can also be used as a context manager::

        with Experiment("my_run") as exp:
            ...
    """

    def __init__(self, name: str, base_dir: str = "sim_results", h5_name: str = "results.h5"):
        """
        Create the experiment directory and open the HDF5 file for writing.

        Parameters
        ----------
        name:
            Human-readable experiment label appended after the timestamp.
        base_dir:
            Root directory under which the experiment folder is created.
        h5_name:
            Name of the HDF5 results file (default ``"results.h5"``).
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = Path(base_dir) / f"{timestamp}_{name}"
        self.path.mkdir(parents=True, exist_ok=True)

        self.h5_path = self.path / h5_name
        self.h5 = h5py.File(self.h5_path, "w")

        self.g_params  = self.h5.require_group("params")
        self.g_sensors = self.h5.require_group("sensors")
        self.g_derived = self.h5.require_group("derived")

        _write_attr(self.h5, "created_at",      datetime.now().isoformat(timespec="seconds"))
        _write_attr(self.h5, "experiment_dir",  str(self.path))

        self.readme = self.path / "README.md"

        print(f"Experiment: {self.path}")
        print(f"  HDF5: {self.h5_path}")

    def close(self) -> None:
        """Flush and close the underlying HDF5 file."""
        try:
            self.h5.flush()
        finally:
            self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ── README ─────────────────────────────────────────────────────────────────

    def log_readme(self, text: str) -> None:
        """Write *text* to ``README.md``, replacing any existing content."""
        self.readme.write_text(text, encoding="utf-8")

    def update_log_readme(
        self,
        status: str,
        runtime_s: float,
        finished_at: str,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Append a status block to ``README.md``.

        Parameters
        ----------
        status:
            Short status string, e.g. ``"ok"`` or ``"failed"``.
        runtime_s:
            Wall-clock runtime in seconds.
        finished_at:
            ISO-8601 finish timestamp.
        error_message:
            Optional error description appended when the run failed.
        """
        content = (
            f"Status: {status}\n"
            f"Runtime: {runtime_s:.2f} seconds\n"
            f"Finished at: {finished_at}"
        )
        if error_message:
            content += f"\nError: {error_message}"

        existing = self.readme.read_text(encoding="utf-8") if self.readme.exists() else ""
        self.readme.write_text(
            (existing + "\n\n" + content) if existing else content,
            encoding="utf-8",
        )

    # ── Parameters ─────────────────────────────────────────────────────────────

    def log_params(self, units: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """
        Persist simulation parameters to the ``/params`` HDF5 group.

        - Scalar values (str, int, float, bool, numpy scalars, Vec3, Vec2,
          CVec2, pybind11 enums) are stored as HDF5 attributes.
        - Array-like values are stored as compressed HDF5 datasets.

        Parameters
        ----------
        units:
            Optional mapping ``{param_name: unit_string}`` that annotates
            each stored value with a ``unit`` attribute.
        **kwargs:
            Parameter names and values to persist.
        """
        for k, v in kwargs.items():
            if _is_scalar(v) or hasattr(v, "name") or hasattr(v, "x") or hasattr(v, "m"):
                _write_attr(self.g_params, k, v)
                if units and k in units:
                    self.g_params.attrs[f"unit:{k}"] = str(units[k])
            else:
                arr = _as_array(v)
                if arr is None:
                    _write_attr(self.g_params, k, str(v))
                else:
                    ds = _write_dataset(self.g_params, k, arr)
                    if units and k in units:
                        ds.attrs["unit"] = str(units[k])
        self.h5.flush()

    # ── Script snapshot ────────────────────────────────────────────────────────

    def log_script(self, file_path: str, out_name: str = "script_snapshot.py") -> None:
        """
        Copy the simulation script into the experiment directory for
        reproducibility.  A minimal reference (filename + original path) is
        stored in the HDF5 file; the full source is kept as a plain ``.py``
        file outside the HDF5.

        Parameters
        ----------
        file_path:
            Absolute path of the script to copy (typically ``__file__``).
        out_name:
            Destination filename inside the experiment directory.

        Raises
        ------
        FileNotFoundError:
            If *file_path* does not exist.
        """
        file_path = str(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Script file not found: {file_path}")
        shutil.copy(file_path, self.path / out_name)
        _write_attr(self.h5, "script_snapshot_file", out_name)
        _write_attr(self.h5, "script_original_path", file_path)
        self.h5.flush()

    # ── Derived arrays ─────────────────────────────────────────────────────────

    def save_derived(
        self,
        key: str,
        arr: np.ndarray,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a post-processed array under ``/derived/<key>``.

        Nested keys are supported with ``/`` separators::

            exp.save_derived("farfield_cbs/S0_coh", S0_coh)

        Parameters
        ----------
        key:
            Path within the ``derived`` group, e.g. ``"farfield_cbs/S0_coh"``.
        arr:
            Array to store.
        attrs:
            Optional dict of metadata to attach as HDF5 dataset attributes.
        """
        parts = key.strip("/").split("/")
        *grp_parts, name = parts
        g = self.g_derived
        for p in grp_parts:
            g = g.require_group(p)

        ds = _write_dataset(g, name, np.asarray(arr))
        if attrs:
            for ak, av in attrs.items():
                try:
                    ds.attrs[ak] = _clean_scalar(av) if _is_scalar(av) else str(av)
                except Exception:
                    ds.attrs[ak] = str(av)
        self.h5.flush()

    # ── Sensor persistence ─────────────────────────────────────────────────────

    def save_sensor(self, sensor: Any, name: str) -> None:
        """
        Persist a sensor's data to ``/sensors/<name>/`` in the HDF5 file.

        Each sensor is stored under two sub-groups:

        - ``meta/`` — type name, grid dimensions, filter settings (HDF5 attrs).
        - ``data/`` — Stokes / field / record arrays (HDF5 datasets).

        To add support for a new sensor type, insert an ``elif`` branch
        below that writes the relevant attributes and arrays.

        Parameters
        ----------
        sensor:
            A sensor object returned by ``SensorsGroup.add_detector()``.
        name:
            Key used to identify this sensor inside the HDF5 file.

        Raises
        ------
        TypeError:
            If the sensor type is not yet supported.
        """
        g      = self.g_sensors.require_group(name)
        g_meta = g.require_group("meta")
        g_data = g.require_group("data")

        # ── Common base attributes (all sensor types) ──────────────────────────
        _write_attr(g_meta, "type", sensor.__class__.__name__)
        _write_attr(g_meta, "id", str(sensor.id))
        _write_attr(g_meta, "origin", sensor.origin)
        _write_attr(g_meta, "normal", sensor.normal)
        _write_attr(g_meta, "backward_normal", sensor.backward_normal)
        _write_attr(g_meta, "n_polarization", sensor.n_polarization)
        _write_attr(g_meta, "m_polarization", sensor.m_polarization)
        _write_attr(g_meta, "hits", sensor.hits)
        _write_attr(g_meta, "absorb_photons", sensor.absorb_photons)
        _write_attr(g_meta, "estimator_enabled", sensor.estimator_enabled)
        _write_attr(g_meta, "filter_theta_enabled", sensor.filter_theta_enabled)
        _write_attr(g_meta, "filter_theta_min", sensor.filter_theta_min)
        _write_attr(g_meta, "filter_theta_max", sensor.filter_theta_max)
        _write_attr(g_meta, "filter_phi_enabled", sensor.filter_phi_enabled)
        _write_attr(g_meta, "filter_phi_min", sensor.filter_phi_min)
        _write_attr(g_meta, "filter_phi_max", sensor.filter_phi_max)
        _write_attr(g_meta, "filter_position_enabled", sensor.filter_position_enabled)
        _write_attr(g_meta, "filter_x_min", sensor.filter_x_min)
        _write_attr(g_meta, "filter_x_max", sensor.filter_x_max)
        _write_attr(g_meta, "filter_y_min", sensor.filter_y_min)
        _write_attr(g_meta, "filter_y_max", sensor.filter_y_max)

        

        t = sensor.__class__.__name__

        # ── PhotonRecordSensor ─────────────────────────────────────────────────
        if t == "PhotonRecordSensor":
            rec = sensor.recorded_photons
            n   = len(rec)
            _write_attr(g_meta, "n_records", n)

            if n > 0:
                rg = g.require_group("records")
                events = np.array([r.events                          for r in rec], dtype=np.int32)
                weight = np.array([r.weight                          for r in rec], dtype=np.float64)
                depth  = np.array([r.penetration_depth               for r in rec], dtype=np.float64)
                t0     = np.array([r.launch_time                     for r in rec], dtype=np.float64)
                t1     = np.array([r.arrival_time                    for r in rec], dtype=np.float64)
                r1     = np.array([[r.position_first_scattering.x,
                                    r.position_first_scattering.y,
                                    r.position_first_scattering.z]   for r in rec], dtype=np.float64)
                rn     = np.array([[r.position_last_scattering.x,
                                    r.position_last_scattering.y,
                                    r.position_last_scattering.z]    for r in rec], dtype=np.float64)
                Ef     = np.array([[r.polarization_forward.m,
                                    r.polarization_forward.n]        for r in rec], dtype=np.complex128)
                Er     = np.array([[r.polarization_reverse.m,
                                    r.polarization_reverse.n]        for r in rec], dtype=np.complex128)

                _write_dataset(rg, "events",            events)
                _write_dataset(rg, "weight",            weight)
                _write_dataset(rg, "penetration_depth", depth)
                _write_dataset(rg, "launch_time",       t0)
                _write_dataset(rg, "arrival_time",      t1)
                _write_dataset(rg, "r_first",           r1)
                _write_dataset(rg, "r_last",            rn)
                _write_dataset(rg, "E_forward",         Ef)
                _write_dataset(rg, "E_reverse",         Er)

        # ── PlanarFieldSensor ──────────────────────────────────────────────────
        elif t == "PlanarFieldSensor":
            for k in ["N_x", "N_y", "dx", "dy", "len_x", "len_y"]:
                _write_attr(g_meta, k, getattr(sensor, k))
            _write_dataset(g_data, "Ex", _as_array(sensor.Ex))
            _write_dataset(g_data, "Ey", _as_array(sensor.Ey))

        # ── PlanarFluenceSensor ────────────────────────────────────────────────
        elif t == "PlanarFluenceSensor":
            for k in ["N_x", "N_y", "N_t", "dx", "dy", "dt", "len_x", "len_y", "len_t"]:
                _write_attr(g_meta, k, getattr(sensor, k))
            # S*_t may be a list of matrices → stacked to (Nt, Nx, Ny)
            _write_dataset(g_data, "S0_t", _as_array(sensor.S0_t))
            _write_dataset(g_data, "S1_t", _as_array(sensor.S1_t))
            _write_dataset(g_data, "S2_t", _as_array(sensor.S2_t))
            _write_dataset(g_data, "S3_t", _as_array(sensor.S3_t))

        # ── PlanarCBSSensor ────────────────────────────────────────────────────
        elif t == "PlanarCBSSensor":
            for k in ["N_x", "N_y", "dx", "dy", "len_x", "len_y"]:
                _write_attr(g_meta, k, getattr(sensor, k))
            _write_dataset(g_data, "S0", _as_array(sensor.S0))
            _write_dataset(g_data, "S1", _as_array(sensor.S1))
            _write_dataset(g_data, "S2", _as_array(sensor.S2))
            _write_dataset(g_data, "S3", _as_array(sensor.S3))

        # ── FarFieldFluenceSensor ──────────────────────────────────────────────
        elif t == "FarFieldFluenceSensor":
            for k in ["N_theta", "N_phi", "theta_max", "phi_max", "dtheta", "dphi"]:
                _write_attr(g_meta, k, getattr(sensor, k))
            _write_dataset(g_data, "S0", _as_array(sensor.S0))
            _write_dataset(g_data, "S1", _as_array(sensor.S1))
            _write_dataset(g_data, "S2", _as_array(sensor.S2))
            _write_dataset(g_data, "S3", _as_array(sensor.S3))

        # ── FarFieldCBSSensor ──────────────────────────────────────────────────
        elif t == "FarFieldCBSSensor":
            for k in [
                "N_theta", "N_phi", "theta_max", "phi_max", "dtheta", "dphi",
                "theta_pp_max", "theta_stride", "phi_stride",
            ]:
                if hasattr(sensor, k):
                    _write_attr(g_meta, k, getattr(sensor, k))

            _write_dataset(g_data, "S0_coh",   _as_array(sensor.S0_coh))
            _write_dataset(g_data, "S1_coh",   _as_array(sensor.S1_coh))
            _write_dataset(g_data, "S2_coh",   _as_array(sensor.S2_coh))
            _write_dataset(g_data, "S3_coh",   _as_array(sensor.S3_coh))
            _write_dataset(g_data, "S0_incoh", _as_array(sensor.S0_incoh))
            _write_dataset(g_data, "S1_incoh", _as_array(sensor.S1_incoh))
            _write_dataset(g_data, "S2_incoh", _as_array(sensor.S2_incoh))
            _write_dataset(g_data, "S3_incoh", _as_array(sensor.S3_incoh))

        # ── StatisticsSensor ───────────────────────────────────────────────────
        elif t == "StatisticsSensor":
            for k in [
                "events_histogram_bins_set", "max_events",
                "theta_histogram_bins_set", "min_theta", "max_theta", "n_bins_theta", "dtheta",
                "phi_histogram_bins_set", "min_phi", "max_phi", "n_bins_phi", "dphi",
                "depth_histogram_bins_set", "max_depth", "n_bins_depth", "ddepth",
                "time_histogram_bins_set", "max_time", "n_bins_time", "dtime",
                "weight_histogram_bins_set", "max_weight", "n_bins_weight", "dweight",
            ]:
                _write_attr(g_meta, k, getattr(sensor, k))
            for name_ in [
                "events_histogram", "theta_histogram", "phi_histogram",
                "depth_histogram", "time_histogram", "weight_histogram"
            ]:
                seq = getattr(sensor, name_)

                try:
                    if len(seq) == 0:
                        _write_attr(g_meta, f"skipped:{name_}", "empty (bins not set?)")
                        continue
                except Exception:
                    pass

                arr = np.asarray(list(seq), dtype=np.int64)
                _write_dataset(g_data, name_, arr)

        else:
            raise TypeError(
                f"Unsupported sensor type: '{t}'. "
                "Add a corresponding elif block in Experiment.save_sensor()."
            )

        self.h5.flush()


# ══════════════════════════════════════════════════════════════════════════════
#  ResultsLoader
# ══════════════════════════════════════════════════════════════════════════════

class ResultsLoader:
    """
    Read-only accessor for a previously saved ``results.h5`` file.

    On construction, all scalar parameters stored in ``/params`` are loaded
    into :attr:`params` as a plain dict.  Array parameters and all sensor /
    derived data are accessed lazily through the helper methods or via direct
    HDF5 path indexing.

    Usage::

        with ResultsLoader("/data/results/2026-02-19_my_run") as loader:
            S0 = loader.sensor_data("far_field_cbs", "S0_coh")
            n_photons = loader.params["n_photons"]
    """

    def __init__(self, exp_dir: Union[str, Path], h5_name: str = "results.h5"):
        """
        Open the HDF5 file for reading and pre-load all scalar parameters.

        Parameters
        ----------
        exp_dir:
            Path to the experiment directory (the one containing ``results.h5``).
        h5_name:
            Name of the HDF5 file (default ``"results.h5"``).

        Raises
        ------
        FileNotFoundError:
            If ``<exp_dir>/<h5_name>`` does not exist.
        """
        self.path    = Path(exp_dir)
        self.h5_path = self.path / h5_name
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        self.h5 = h5py.File(self.h5_path, "r")

        # Pre-load all scalar attrs and array datasets from /params
        self.params: Dict[str, Any] = {}
        g = self.h5.get("params", None)
        if g is not None:
            for k, v in g.attrs.items():
                self.params[k] = v
            for k in g.keys():
                obj = g[k]
                if isinstance(obj, h5py.Dataset):
                    self.params[k] = np.asarray(obj)

    def close(self) -> None:
        """Close the underlying HDF5 file."""
        self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __getitem__(self, key: str) -> np.ndarray:
        """
        Direct HDF5 path access::

            loader["sensors/far_field_cbs/data/S0_coh"]
        """
        return np.asarray(self.h5[key])

    # ── Convenience accessors ──────────────────────────────────────────────────

    def sensor_data(self, sensor_name: str, dataset: str) -> np.ndarray:
        """
        Load a dataset from ``/sensors/<sensor_name>/data/<dataset>``.

        Parameters
        ----------
        sensor_name:
            Name used when the sensor was saved with
            :py:meth:`Experiment.save_sensor`.
        dataset:
            Dataset key, e.g. ``"S0_coh"``.
        """
        return np.asarray(self.h5[f"sensors/{sensor_name}/data/{dataset}"])

    def sensor_meta(self, sensor_name: str) -> Dict[str, Any]:
        """
        Return all metadata attributes of a sensor as a plain dict.

        Parameters
        ----------
        sensor_name:
            Name used when the sensor was saved.
        """
        g = self.h5[f"sensors/{sensor_name}/meta"]
        return dict(g.attrs.items())

    def derived(self, key: str) -> np.ndarray:
        """
        Load a post-processed array from ``/derived/<key>``.

        Parameters
        ----------
        key:
            Path within the ``derived`` group, e.g. ``"farfield_cbs/S0_coh"``.
        """
        return np.asarray(self.h5[f"derived/{key}"])
