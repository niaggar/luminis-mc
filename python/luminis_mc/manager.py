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
PhotonRecordSensor, PlanarFieldSensor, PlanarFluenceSensor,
FarFieldCBSSensor, StatisticsSensor.
The mapping from sensor attributes to HDF5 groups is data-driven: see the
``SENSOR_SCHEMAS`` registry in :mod:`luminis_mc.schema`.  To add a new sensor
type, add one entry there — no code in this module needs to change.

Parameters are auto-captured from the simulation objects with
:func:`capture_params` and stored both as a typed JSON blob
(``/params`` attribute ``params_json``, reloaded as :class:`SimParams`) and as
flat scalars under ``/params`` for quick inspection.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING, Union

import numpy as np
import h5py

if TYPE_CHECKING:
    # For static typing only — avoids a hard runtime dependency on the compiled
    # extension in this pure-Python persistence module.
    from ._core import ScatteringMedium

from . import schema
from .records import (
    SimParams, RunParams, LaserParams, MediumParams, LayerParams,
    RESULT_BUILDERS,
    FarFieldCBSResult, PlanarFluenceResult, PlanarFieldResult,
    StatisticsResult, PhotonRecordsResult, AbsorptionResult,
    FarFieldCBSProcessedResult, PlanarFluenceProcessedResult, PlanarFieldProcessedResult,
    _to_numpy, _py,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def on_progress(done, total):
    pct = done * 100 // total
    print(f"\rSimulation progress: {pct}%", end="", flush=True)
    if done == total:
        print()


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
#  Parameter capture
# ══════════════════════════════════════════════════════════════════════════════

def _medium_params(medium) -> MediumParams:
    """Build :class:`MediumParams` from a (bound) scattering medium object."""
    return MediumParams(
        kind=medium.__class__.__name__,
        mu_s=float(medium.mu_s),
        mu_a=float(medium.mu_a),
        mu_t=float(medium.mu_t),
        n_particle=float(medium.n_particle),
        n_medium=float(medium.n_medium),
        wavelength=float(medium.wavelength),
        radius=(float(getattr(medium, "radius")) if hasattr(medium, "radius") else None),
        mean_free_path=(float(getattr(medium, "mean_free_path"))
                        if hasattr(medium, "mean_free_path") else None),
    )


def _layer_params(L) -> LayerParams:
    """
    Build a :class:`LayerParams` from a bound layer.

    A ``MixtureLayer`` exposes ``species`` / ``number_densities`` and aggregate
    totals; a homogeneous ``SampleLayer`` exposes a single ``medium``.
    """
    if hasattr(L, "species") and not hasattr(L, "medium"):
        species = list(L.species)
        return LayerParams(
            z_min=float(L.z_min), z_max=float(L.z_max), kind="mixture",
            species=[_medium_params(s) for s in species],
            number_densities=[float(n) for n in L.number_densities],
            mu_s_total=float(L.mu_s_total),
            mu_a_total=float(L.mu_a_total),
            mu_t_total=float(L.mu_t_total),
            mfp_total=float(L.mfp_total),
        )
    return LayerParams(
        z_min=float(L.z_min), z_max=float(L.z_max), kind="homogeneous",
        medium=_medium_params(L.medium),
    )


def capture_params(config, extra: Optional[Dict[str, Any]] = None) -> SimParams:
    """
    Auto-capture a typed :class:`SimParams` snapshot from a ``SimConfig``.

    Reads the run controls, laser, host medium, and every sample layer directly
    from the bound objects, so the user never re-lists parameters by hand.
    Derived / physical quantities that do not live on the C++ objects (``g``,
    ``l*``, ``theta_coherent``, ...) can be supplied via *extra* — see
    :func:`derived_quantities`.

    Parameters
    ----------
    config:
        A ``SimConfig`` with ``sample`` and ``laser`` set.
    extra:
        Optional mapping of derived quantities stored under ``SimParams.extra``.

    Raises
    ------
    ValueError:
        If ``config.sample`` or ``config.laser`` is not set.
    """
    if getattr(config, "sample", None) is None:
        raise ValueError("capture_params: config.sample is not set.")
    if getattr(config, "laser", None) is None:
        raise ValueError("capture_params: config.laser is not set.")

    sample = config.sample
    laser = config.laser
    pol = laser.polarization

    run = RunParams(
        n_photons=int(getattr(config, "n_photons", 0)),
        n_threads=int(getattr(config, "n_threads", 1)),
        seed=int(getattr(config, "seed", 0)),
        max_events=int(getattr(config, "MAX_EVENTS", 0)),
        track_reverse_paths=bool(getattr(config, "track_reverse_paths", False)),
    )
    laser_p = LaserParams(
        source_type=str(laser.source_type.name),
        wavelength=float(laser.wavelength),
        sigma=float(laser.sigma),
        m_state=complex(pol.m),
        n_state=complex(pol.n),
    )
    layers = [_layer_params(L) for L in sample.layers]
    return SimParams(
        run=run,
        laser=laser_p,
        sample_n_medium=float(sample.refractive_index),
        layers=layers,
        extra=dict(extra) if extra else {},
    )


def derived_quantities(medium, volume_fraction: float) -> Dict[str, Any]:
    """
    Compute the standard set of derived physical quantities for *medium*.

    Centralises the boilerplate that every script repeated by hand: scattering
    efficiency, anisotropy ``g``, scattering mean free path ``l_s``, transport
    mean free path ``l*``, CBS cone width ``theta_coherent = 1/(k·l*)``, size
    parameter, and relative index.  The returned dict is meant to be passed as
    ``extra=`` to :func:`capture_params` / :py:meth:`Experiment.save_params`.

    Parameters
    ----------
    medium:
        An ``RGDMedium`` / ``MieMedium`` (must expose ``radius``,
        ``n_particle``, ``n_medium``, ``wavelength`` and ``phase_function``).
    volume_fraction:
        Particle volume fraction used to scale the mean free path.
    """
    phase = medium.phase_function
    q_sca = float(phase.scattering_efficiency())
    g = float(phase.get_anisotropy_factor()[0])
    radius = float(medium.radius)
    n_medium = float(medium.n_medium)
    n_particle = float(medium.n_particle)
    wavelength = float(medium.wavelength)

    mean_free_path = (4.0 * radius) / (3.0 * volume_fraction * q_sca)
    transport_mean_free_path = mean_free_path / (1.0 - g)
    k_medium = 2.0 * np.pi * n_medium / wavelength

    return {
        "volume_fraction": float(volume_fraction),
        "scattering_efficiency": q_sca,
        "anisotropy_g": g,
        "mean_free_path": mean_free_path,
        "transport_mean_free_path": transport_mean_free_path,
        "theta_coherent": 1.0 / (k_medium * transport_mean_free_path),
        "size_parameter": 2.0 * np.pi * radius * n_medium / wavelength,
        "m_relative": n_particle / n_medium,
    }


def derived_quantities_mixture(
    species: Sequence["ScatteringMedium"],
    number_densities: Sequence[float],
) -> Dict[str, Any]:
    """
    Derived physical quantities for a multi-species (mixture) layer.

    Computes per-species ``μ_s^(i) = n_i · σ_s^(i)`` (from each species' phase
    function cross-section) and the mixture aggregates used to interpret the CBS
    cone:

    - effective anisotropy  ``g_eff = Σ μ_s^(i) g_i / Σ μ_s^(i)``
    - scattering mean free path  ``l_s = 1 / Σ μ_s^(i)``
    - transport mean free path   ``l* = 1 / Σ μ_s^(i) (1 − g_i)``
    - CBS cone width             ``theta_coherent = 1 / (k · l*)``

    Returned dict is meant to be passed as ``extra=`` to
    :func:`capture_params` / :py:meth:`Experiment.save_params`.

    Parameters
    ----------
    species:
        Iterable of scattering media (each must expose ``phase_function``,
        ``n_medium`` and ``wavelength``).
    number_densities:
        Number density ``n_i`` [1/mm³] for each species (same order/length).
    """
    species = list(species)
    number_densities = [float(n) for n in number_densities]
    if not species:
        raise ValueError("derived_quantities_mixture: species list is empty.")
    if len(species) != len(number_densities):
        raise ValueError("derived_quantities_mixture: species and number_densities differ in length.")

    # Wave number in the host medium. Use the medium's own ``k`` (= 2π·n_medium/λ),
    # which the C++ ctor sets on the base object; the bound ``wavelength`` field is
    # shadowed by the concrete media and is not reliable here.
    k_medium = float(species[0].k)

    mu_s_i: List[float] = []
    g_i: List[float] = []
    for sp, n in zip(species, number_densities):
        sigma = float(sp.phase_function.scattering_cross_section())
        mu_s_i.append(n * sigma)
        g_i.append(float(sp.phase_function.get_anisotropy_factor()[0]))

    mu_s_total = float(sum(mu_s_i))
    if mu_s_total <= 0.0:
        raise ValueError("derived_quantities_mixture: total mu_s is not positive "
                         "(check number densities and phase-function cross-sections).")

    mu_s_reduced = float(sum(ms * (1.0 - g) for ms, g in zip(mu_s_i, g_i)))
    g_eff = float(sum(ms * g for ms, g in zip(mu_s_i, g_i)) / mu_s_total)
    l_s = 1.0 / mu_s_total
    l_star = (1.0 / mu_s_reduced) if mu_s_reduced > 0.0 else float("inf")

    return {
        "mixture": True,
        "n_species": len(species),
        "number_densities": number_densities,
        "mu_s_per_species": mu_s_i,
        "anisotropy_g_per_species": g_i,
        "mu_s_total": mu_s_total,
        "anisotropy_g_eff": g_eff,
        "scattering_mean_free_path": l_s,
        "transport_mean_free_path": l_star,
        "theta_coherent": (1.0 / (k_medium * l_star)) if np.isfinite(l_star) else 0.0,
    }


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

    def __init__(self, name: str, base_dir: str = "sim_results", h5_name: str = "results.h5", timestamped: bool = True):
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
        timestamped:
            When ``True`` (default) the folder is prefixed with a
            ``YYYY-MM-DD_HH-MM-SS`` timestamp, producing a unique directory
            per run.  Set to ``False`` for quick test runs: the folder will
            simply be ``<base_dir>/<name>/`` and any previous results at
            that path are overwritten.
        """
        if timestamped:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.path = Path(base_dir) / f"{timestamp}_{name}"
        else:
            self.path = Path(base_dir) / name
        
        if self.path.exists():
            shutil.rmtree(self.path)
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

    def save_params(
        self,
        config,
        *,
        extra: Optional[Dict[str, Any]] = None,
        units: Optional[Dict[str, str]] = None,
    ) -> SimParams:
        """
        Auto-capture and persist a typed :class:`SimParams` from a ``SimConfig``.

        The full parameter tree is stored as JSON in the ``params_json``
        attribute of ``/params`` (reloaded as a typed :class:`SimParams` by
        :py:attr:`ResultsLoader.params`).  A flat projection of the most useful
        scalars is also written under ``/params`` via :py:meth:`log_params` so
        the file stays inspectable with ``h5dump`` / ``h5ls``.

        Parameters
        ----------
        config:
            A ``SimConfig`` with ``sample`` and ``laser`` set.
        extra:
            Derived quantities to record (see :func:`derived_quantities`).
        units:
            Optional ``{name: unit}`` annotations for the flat scalars.

        Returns
        -------
        SimParams
            The captured snapshot (also returned for convenience / logging).
        """
        params = capture_params(config, extra)

        self.g_params.attrs["params_json"] = params.to_json()

        # Flat projection for quick inspection.
        flat: Dict[str, Any] = {
            "n_photons": params.run.n_photons,
            "n_threads": params.run.n_threads,
            "seed": params.run.seed,
            "max_events": params.run.max_events,
            "track_reverse_paths": params.run.track_reverse_paths,
            "laser_source_type": params.laser.source_type,
            "wavelength": params.laser.wavelength,
            "laser_sigma": params.laser.sigma,
            "sample_n_medium": params.sample_n_medium,
            "n_layers": len(params.layers),
        }
        if params.layers:
            top_layer = params.layers[0]
            flat["layer_kind"] = top_layer.kind
            if top_layer.kind == "mixture":
                # Aggregate coefficients for the mixture; the per-species detail
                # lives in params_json (and the *species* list on the layer).
                flat.update({
                    "medium_kind": "mixture",
                    "n_species": len(top_layer.species or []),
                    "mu_s": top_layer.mu_s_total,
                    "mu_a": top_layer.mu_a_total,
                    "mu_t": top_layer.mu_t_total,
                    "mfp_total": top_layer.mfp_total,
                })
                if top_layer.species:
                    flat["n_medium"] = top_layer.species[0].n_medium
            else:
                top = top_layer.medium
                flat.update({
                    "medium_kind": top.kind,
                    "mu_s": top.mu_s,
                    "mu_a": top.mu_a,
                    "mu_t": top.mu_t,
                    "n_particle": top.n_particle,
                    "n_medium": top.n_medium,
                })
                if top.radius is not None:
                    flat["radius"] = top.radius
                if top.mean_free_path is not None:
                    flat["mean_free_path"] = top.mean_free_path
        for k, v in params.extra.items():
            if _is_scalar(v):
                flat[k] = v
        self.log_params(units=units, **flat)
        return params

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

    # ── Absorption persistence ─────────────────────────────────────────────────

    def save_absorption(self, absorption: Any, n_photons: int, name: str = "absorption") -> None:
        """
        Persist an ``Absorption`` recorder to ``/absorption/<name>/`` in the HDF5 file.

        Stores the grid configuration as metadata attributes and each time-slice
        matrix as a dataset.  For time-integrated recordings (``d_t == 0``)
        only a single slice ``time_slice_0`` is written.

        Parameters
        ----------
        absorption:
            An ``Absorption`` object from the simulation.
        name:
            Key used to identify this absorption recorder inside the HDF5 file
            (default ``"absorption"``).
        """
        g_abs = self.h5.require_group("absorption")
        g = g_abs.require_group(name)
        g_meta = g.require_group("meta")
        g_data = g.require_group("data")

        # ── Grid configuration ─────────────────────────────────────────────────
        _write_attr(g_meta, "radius", absorption.radius)
        _write_attr(g_meta, "depth", absorption.depth)
        _write_attr(g_meta, "d_r", absorption.d_r)
        _write_attr(g_meta, "d_z", absorption.d_z)
        _write_attr(g_meta, "d_t", absorption.d_t)
        _write_attr(g_meta, "t_max", absorption.t_max)
        _write_attr(g_meta, "n_t", absorption.n_t)

        # ── Time-slice data (slice 0 is always integrated) ───────────────────
        for i, ts in enumerate(absorption.time_slices):
            image = absorption.get_absorption_image(n_photons, i)
            arr_image = _as_array(image)
            arr = _as_array(ts)

            if arr is not None:
                _write_dataset(g_data, f"time_slice_{i}", arr)
                _write_dataset(g_data, f"time_slice_{i}_image", arr_image)

        self.h5.flush()

    # ── Sensor persistence ─────────────────────────────────────────────────────

    def save_sensors(self, sensors: Dict[str, Any]) -> None:
        """
        Persist a mapping ``{name: sensor}`` in one call.

        Convenience wrapper around :py:meth:`save_sensor` for saving a whole
        ``SensorsGroup`` worth of detectors::

            exp.save_sensors({"farfield_cbs": det, "statistics": stats})
        """
        for name, sensor in sensors.items():
            self.save_sensor(sensor, name)

    def save_sensor(self, sensor: Any, name: str) -> None:
        """
        Persist a sensor's data to ``/sensors/<name>/`` in the HDF5 file.

        Each sensor is stored under two sub-groups:

        - ``meta/`` — type name, grid dimensions, filter settings (HDF5 attrs).
        - ``data/`` — Stokes / field / record arrays (HDF5 datasets).

        The attribute→group mapping is data-driven via the ``SENSOR_SCHEMAS``
        registry in :mod:`luminis_mc.schema`.  To support a new sensor type, add
        a ``SensorSchema`` entry there — this method does not change.

        Parameters
        ----------
        sensor:
            A sensor object returned by ``SensorsGroup.add_detector()``.
        name:
            Key used to identify this sensor inside the HDF5 file.

        Raises
        ------
        KeyError:
            If the sensor type has no registered schema.
        """
        sch = schema.get_schema(sensor.__class__.__name__)

        g      = self.g_sensors.require_group(name)
        g_meta = g.require_group("meta")
        g_data = g.require_group("data")

        # ── Type tag + common base attributes ──────────────────────────────────
        _write_attr(g_meta, "type", sensor.__class__.__name__)
        for attr in schema.COMMON_META:
            if hasattr(sensor, attr):
                _write_attr(g_meta, attr, getattr(sensor, attr))

        # ── Type-specific scalar metadata ──────────────────────────────────────
        for attr in sch.meta_attrs:
            if hasattr(sensor, attr):
                _write_attr(g_meta, attr, getattr(sensor, attr))

        # ── Data: custom encoder or generic array dispatch ─────────────────────
        if sch.encoder is not None:
            sch.encoder(sensor, g, g_meta, g_data, _write_dataset, _write_attr)
        else:
            for attr in sch.data_attrs:
                arr = _as_array(getattr(sensor, attr))
                if arr is None or arr.size == 0:
                    _write_attr(g_meta, f"skipped:{attr}", "empty (bins not set?)")
                    continue
                _write_dataset(g_data, attr, arr)

        self.h5.flush()

    # ── Post-processed results ───────────────────────────────────────────────────

    def save_processed(
        self,
        name: str,
        processed: Any,
        *,
        sensor: Any = None,
        theta: Optional[np.ndarray] = None,
        phi: Optional[np.ndarray] = None,
    ) -> None:
        """
        Persist a post-processed result object under ``/derived/<name>/``.

        Dispatches on the type returned by the ``postprocess_*`` functions:

        - ``FarFieldCBSProcessed`` → ``coherent/{s0..s3}`` and
          ``incoherent/{s0..s3}`` each stacked over time bins to shape
          ``(N_t, N_theta, N_phi)``, plus ``axes/theta`` and ``axes/phi``.
        - ``PlanarFluenceProcessed`` → ``{s0..s3}`` of shape ``(N_t, N_x, N_y)``.
        - ``PlanarFieldProcessed`` → ``Ex``, ``Ey``.

        Replaces the manual per-time-bin string-keyed save loops.

        Parameters
        ----------
        name:
            Key under ``/derived`` (e.g. ``"farfield_cbs"``).
        processed:
            Object returned by ``postprocess_farfield_cbs`` /
            ``postprocess_planar_fluence`` / ``postprocess_planar_field``.
        sensor:
            Originating sensor; used to derive the ``theta``/``phi`` axes for
            CBS results when *theta*/*phi* are not given explicitly.
        theta, phi:
            Explicit axes (override *sensor*-derived ones).
        """
        cls = processed.__class__.__name__

        if cls == "FarFieldCBSProcessed":
            coh, inc = processed.coherent, processed.incoherent
            for comp, items in (("coherent", coh), ("incoherent", inc)):
                for s in ("S0", "S1", "S2", "S3"):
                    stack = np.stack([_to_numpy(getattr(it, s)) for it in items], axis=0)
                    self.save_derived(f"{name}/{comp}/{s.lower()}", stack)

            if theta is None and sensor is not None:
                theta = np.linspace(0.0, sensor.theta_max, sensor.N_theta)
            if phi is None and sensor is not None:
                phi = np.linspace(0.0, sensor.phi_max, sensor.N_phi)
            if theta is not None:
                self.save_derived(f"{name}/axes/theta", np.asarray(theta))
            if phi is not None:
                self.save_derived(f"{name}/axes/phi", np.asarray(phi))

        elif cls == "PlanarFluenceProcessed":
            for s in ("S0", "S1", "S2", "S3"):
                stack = np.stack([_to_numpy(m) for m in getattr(processed, s)], axis=0)
                self.save_derived(f"{name}/{s.lower()}", stack)

        elif cls == "PlanarFieldProcessed":
            self.save_derived(f"{name}/Ex", _to_numpy(processed.Ex))
            self.save_derived(f"{name}/Ey", _to_numpy(processed.Ey))

        else:
            raise TypeError(
                f"save_processed: unsupported processed type '{cls}'. "
                "Supported: FarFieldCBSProcessed, PlanarFluenceProcessed, PlanarFieldProcessed."
            )


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

        # Flat projection of /params (scalars + array datasets). The typed
        # snapshot is available via the `params` property.
        self.params_flat: Dict[str, Any] = {}
        self._params_json: Optional[str] = None
        g = self.h5.get("params", None)
        if g is not None:
            for k, v in g.attrs.items():
                if k == "params_json":
                    self._params_json = _py(v)
                    continue
                self.params_flat[k] = _py(v)
            for k in g.keys():
                obj = g[k]
                if isinstance(obj, h5py.Dataset):
                    self.params_flat[k] = np.asarray(obj)

        self._params_cache: Optional[SimParams] = None

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

    # ── Typed parameters ───────────────────────────────────────────────────────

    @property
    def params(self) -> SimParams:
        """
        The typed :class:`SimParams` snapshot reconstructed from ``params_json``.

        Raises
        ------
        KeyError:
            If the file has no ``params_json`` (e.g. params saved only via the
            legacy flat :py:meth:`Experiment.log_params`).  Use
            :py:attr:`params_flat` for that case.
        """
        if self._params_cache is None:
            if self._params_json is None:
                raise KeyError(
                    "No typed params found (params_json missing). "
                    "Use `.params_flat` for the flat scalar dict."
                )
            self._params_cache = SimParams.from_json(self._params_json)
        return self._params_cache

    # ── Typed sensor accessors ──────────────────────────────────────────────────

    def sensor_names(self) -> List[str]:
        """Return the names of all saved sensors."""
        g = self.h5.get("sensors", None)
        return list(g.keys()) if g is not None else []

    def _read_meta(self, name: str) -> Dict[str, Any]:
        g = self.h5[f"sensors/{name}/meta"]
        return {k: _py(v) for k, v in g.attrs.items()}

    def _read_data(self, name: str) -> Dict[str, np.ndarray]:
        g = self.h5.get(f"sensors/{name}/data", None)
        if g is None:
            return {}
        return {k: np.asarray(g[k]) for k in g.keys()}

    def _read_group_arrays(self, path: str) -> Dict[str, np.ndarray]:
        g = self.h5.get(path, None)
        if g is None:
            return {}
        return {k: np.asarray(g[k]) for k in g.keys()}

    def _check_type(self, name: str, expected: str) -> Dict[str, Any]:
        meta = self._read_meta(name)
        actual = meta.get("type")
        if actual != expected:
            raise TypeError(
                f"Sensor '{name}' has type '{actual}', expected '{expected}'."
            )
        return meta

    def sensor(self, name: str) -> Any:
        """
        Load a sensor by name, returning the typed result for its saved type.

        The concrete return type is one of ``FarFieldCBSResult``,
        ``PlanarFluenceResult``, ``PlanarFieldResult``, ``StatisticsResult`` or
        ``PhotonRecordsResult``.  For static typing, prefer the type-specific
        accessors (:py:meth:`far_field_cbs`, etc.).
        """
        meta = self._read_meta(name)
        t = meta.get("type")
        builder = RESULT_BUILDERS.get(t)
        if builder is None:
            raise TypeError(f"Sensor '{name}' has unsupported type '{t}'.")
        if t == "PhotonRecordSensor":
            return builder(meta, self._read_group_arrays(f"sensors/{name}/records"))
        return builder(meta, self._read_data(name))

    def far_field_cbs(self, name: str = "farfield_cbs") -> FarFieldCBSResult:
        """Load a ``FarFieldCBSSensor`` result."""
        meta = self._check_type(name, "FarFieldCBSSensor")
        return FarFieldCBSResult.from_h5(meta, self._read_data(name))

    def planar_fluence(self, name: str = "planar_fluence") -> PlanarFluenceResult:
        """Load a ``PlanarFluenceSensor`` result."""
        meta = self._check_type(name, "PlanarFluenceSensor")
        return PlanarFluenceResult.from_h5(meta, self._read_data(name))

    def planar_field(self, name: str = "planar_field") -> PlanarFieldResult:
        """Load a ``PlanarFieldSensor`` result."""
        meta = self._check_type(name, "PlanarFieldSensor")
        return PlanarFieldResult.from_h5(meta, self._read_data(name))

    def statistics(self, name: str = "statistics") -> StatisticsResult:
        """Load a ``StatisticsSensor`` result."""
        meta = self._check_type(name, "StatisticsSensor")
        return StatisticsResult.from_h5(meta, self._read_data(name))

    def photon_records(self, name: str = "photon_records") -> PhotonRecordsResult:
        """Load a ``PhotonRecordSensor`` result."""
        meta = self._check_type(name, "PhotonRecordSensor")
        return PhotonRecordsResult.from_h5(meta, self._read_group_arrays(f"sensors/{name}/records"))

    # ── Typed post-processed accessors ──────────────────────────────────────────

    def processed_cbs(self, name: str = "farfield_cbs") -> FarFieldCBSProcessedResult:
        """Load a post-processed CBS result saved with :py:meth:`Experiment.save_processed`."""
        base = f"derived/{name}"
        coh = self._read_group_arrays(f"{base}/coherent")
        inc = self._read_group_arrays(f"{base}/incoherent")
        axes = self._read_group_arrays(f"{base}/axes")
        return FarFieldCBSProcessedResult(
            theta=axes.get("theta", np.empty(0)),
            phi=axes.get("phi", np.empty(0)),
            coh_s0=coh["s0"], coh_s1=coh["s1"], coh_s2=coh["s2"], coh_s3=coh["s3"],
            inc_s0=inc["s0"], inc_s1=inc["s1"], inc_s2=inc["s2"], inc_s3=inc["s3"],
        )

    def processed_fluence(self, name: str) -> PlanarFluenceProcessedResult:
        """Load a post-processed planar-fluence result."""
        d = self._read_group_arrays(f"derived/{name}")
        return PlanarFluenceProcessedResult(S0=d["s0"], S1=d["s1"], S2=d["s2"], S3=d["s3"])

    def processed_field(self, name: str) -> PlanarFieldProcessedResult:
        """Load a post-processed planar-field result."""
        d = self._read_group_arrays(f"derived/{name}")
        return PlanarFieldProcessedResult(Ex=d["Ex"], Ey=d["Ey"])

    # ── Typed absorption accessor ───────────────────────────────────────────────

    def absorption(self, name: str = "absorption") -> AbsorptionResult:
        """
        Load an absorption recorder as a typed :class:`AbsorptionResult`.

        Raw grids and display images are stacked over time bins; slice 0 is the
        time-integrated result (see :py:attr:`AbsorptionResult.total`).
        """
        meta = self.absorption_meta(name)
        g = self.h5.get(f"absorption/{name}/data", None)
        slices: List[np.ndarray] = []
        images: List[np.ndarray] = []
        if g is not None:
            i = 0
            while f"time_slice_{i}" in g:
                slices.append(np.asarray(g[f"time_slice_{i}"]))
                if f"time_slice_{i}_image" in g:
                    images.append(np.asarray(g[f"time_slice_{i}_image"]))
                i += 1
        return AbsorptionResult.from_h5(meta, slices, images)

    # ── Absorption accessors (raw) ──────────────────────────────────────────────

    def absorption_meta(self, name: str = "absorption") -> Dict[str, Any]:
        """
        Return all metadata attributes of a saved absorption recorder as a dict.

        Parameters
        ----------
        name:
            Key used when the absorption was saved with
            :py:meth:`Experiment.save_absorption`.
        """
        g = self.h5[f"absorption/{name}/meta"]
        return dict(g.attrs.items())

    def absorption_data(self, name: str = "absorption", time_index: int = 0) -> np.ndarray:
        """
        Load a time-slice dataset from ``/absorption/<name>/data/time_slice_<time_index>``.

        Parameters
        ----------
        name:
            Key used when the absorption was saved.
        time_index:
            Index of the time slice to load (default ``0``).
        """
        return np.asarray(self.h5[f"absorption/{name}/data/time_slice_{time_index}"])
    
    def absorption_image(self, name: str = "absorption", time_index: int = 0) -> np.ndarray:
        """
        Load the corresponding absorption image from ``/absorption/<name>/data/time_slice_<time_index>_image``.

        Parameters
        ----------
        name:
            Key used when the absorption was saved.
        time_index:
            Index of the time slice to load (default ``0``).
        """
        return np.asarray(self.h5[f"absorption/{name}/data/time_slice_{time_index}_image"])

    def absorption_total_data(self, name: str = "absorption") -> np.ndarray:
        """
        Load the total (time-integrated) absorption grid.

        In current files this is ``/absorption/<name>/data/time_slice_0``.
        For backward compatibility, legacy ``.../data/total`` is also supported.

        Parameters
        ----------
        name:
            Key used when the absorption was saved.
        """
        base = f"absorption/{name}/data"
        path_new = f"{base}/time_slice_0"
        path_old = f"{base}/total"

        if path_new in self.h5:
            return np.asarray(self.h5[path_new])
        if path_old in self.h5:
            return np.asarray(self.h5[path_old])

        raise KeyError(f"No integrated absorption dataset found at {path_new} or {path_old}")

    def absorption_total_image(self, name: str = "absorption") -> np.ndarray:
        """
        Load the total (time-integrated) absorption image.

        In current files this is ``/absorption/<name>/data/time_slice_0_image``.
        For backward compatibility, legacy ``.../data/total_image`` is also supported.

        Parameters
        ----------
        name:
            Key used when the absorption was saved.
        """
        base = f"absorption/{name}/data"
        path_new = f"{base}/time_slice_0_image"
        path_old = f"{base}/total_image"

        if path_new in self.h5:
            return np.asarray(self.h5[path_new])
        if path_old in self.h5:
            return np.asarray(self.h5[path_old])

        raise KeyError(f"No integrated absorption image found at {path_new} or {path_old}")

