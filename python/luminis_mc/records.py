"""
records.py
==========
Typed result and parameter objects for luminis-mc.

These dataclasses are what :class:`luminis_mc.manager.ResultsLoader` hands back,
so the editor / assistant sees concrete field names and types instead of opaque
``Dict[str, Any]`` and string-keyed HDF5 paths.

Two families of types live here:

- **Parameters** (:class:`SimParams` and friends) — a typed, JSON-serialisable
  snapshot of the simulation configuration, auto-captured from the C++ objects
  by :func:`luminis_mc.manager.capture_params`.
- **Results** (:class:`SensorMeta`, ``*Result``) — typed views over the data
  saved for each sensor / post-processed quantity.

The module also owns the ``PhotonRecordSensor`` encoder and registers it on the
:data:`luminis_mc.schema.SENSOR_SCHEMAS` registry (done at import time to avoid
an import cycle with :mod:`luminis_mc.manager`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import schema


# ══════════════════════════════════════════════════════════════════════════════
#  Low-level coercion helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_numpy(x) -> Optional[np.ndarray]:
    """Coerce a binding Matrix/CMatrix, list of matrices, or array to ndarray."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x if x.dtype != object else None
    if hasattr(x, "get_numpy"):
        a = np.asarray(x.get_numpy())
        return a if a.dtype != object else None
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        mats = [_to_numpy(y) for y in x]
        if any(m is None for m in mats):
            a = np.asarray(x)
            return a if a.dtype != object else None
        return np.stack(mats, axis=0)
    a = np.asarray(x)
    return a if a.dtype != object else None


def _py(v) -> Any:
    """Convert an HDF5-read value (numpy scalar/bytes) to a plain Python value."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray):
        return v.item() if v.ndim == 0 else v
    if isinstance(v, np.generic):
        return v.item()
    return v


# ══════════════════════════════════════════════════════════════════════════════
#  Parameters
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MediumParams:
    """Optical parameters of a single scattering medium."""
    kind: str
    mu_s: float
    mu_a: float
    mu_t: float
    n_particle: float
    n_medium: float
    wavelength: float
    radius: Optional[float] = None
    mean_free_path: Optional[float] = None


@dataclass
class LayerParams:
    """
    A single sample layer (z-range + its scattering content).

    Two kinds, discriminated by :attr:`kind`:

    - ``"homogeneous"`` — one medium, stored in :attr:`medium`.
    - ``"mixture"`` — several co-located species, stored in :attr:`species`
      together with their :attr:`number_densities` and the layer's aggregate
      coefficients (``mu_*_total`` / ``mfp_total``).
    """
    z_min: float
    z_max: float
    kind: str = "homogeneous"
    # Homogeneous layer.
    medium: Optional[MediumParams] = None
    # Mixture layer.
    species: Optional[List[MediumParams]] = None
    number_densities: Optional[List[float]] = None
    mu_s_total: Optional[float] = None
    mu_a_total: Optional[float] = None
    mu_t_total: Optional[float] = None
    mfp_total: Optional[float] = None


@dataclass
class LaserParams:
    """Laser source parameters."""
    source_type: str
    wavelength: float
    sigma: float
    m_state: complex
    n_state: complex


@dataclass
class RunParams:
    """Monte-Carlo run controls."""
    n_photons: int
    n_threads: int
    seed: int
    max_events: int
    track_reverse_paths: bool


def _layer_to_dict(L: "LayerParams") -> Dict[str, Any]:
    """Serialise a LayerParams (homogeneous or mixture) to a plain dict."""
    d: Dict[str, Any] = {"z_min": L.z_min, "z_max": L.z_max, "kind": L.kind}
    if L.kind == "mixture":
        d["species"] = [vars(s).copy() for s in (L.species or [])]
        d["number_densities"] = list(L.number_densities or [])
        d["mu_s_total"] = L.mu_s_total
        d["mu_a_total"] = L.mu_a_total
        d["mu_t_total"] = L.mu_t_total
        d["mfp_total"] = L.mfp_total
    else:
        d["medium"] = vars(L.medium).copy() if L.medium is not None else None
    return d


def _layer_from_dict(L: Dict[str, Any]) -> "LayerParams":
    """Rebuild a LayerParams from a dict. Defaults to homogeneous for old files."""
    kind = L.get("kind", "homogeneous")
    if kind == "mixture":
        return LayerParams(
            z_min=L["z_min"], z_max=L["z_max"], kind="mixture",
            species=[MediumParams(**s) for s in L.get("species", [])],
            number_densities=list(L.get("number_densities", [])),
            mu_s_total=L.get("mu_s_total"),
            mu_a_total=L.get("mu_a_total"),
            mu_t_total=L.get("mu_t_total"),
            mfp_total=L.get("mfp_total"),
        )
    medium = L.get("medium")
    return LayerParams(
        z_min=L["z_min"], z_max=L["z_max"], kind="homogeneous",
        medium=MediumParams(**medium) if medium is not None else None,
    )


@dataclass
class SimParams:
    """
    Full typed snapshot of a simulation configuration.

    ``extra`` is the escape hatch for derived / physical quantities that do not
    live on the C++ objects (e.g. anisotropy ``g``, transport mean free path,
    CBS cone width, size parameter, volume fraction).
    """
    run: RunParams
    laser: LaserParams
    sample_n_medium: float
    layers: List[LayerParams]
    extra: Dict[str, Any] = field(default_factory=dict)

    # ── (de)serialisation ───────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run": vars(self.run).copy(),
            "laser": {
                "source_type": self.laser.source_type,
                "wavelength": self.laser.wavelength,
                "sigma": self.laser.sigma,
                "m_state": [self.laser.m_state.real, self.laser.m_state.imag],
                "n_state": [self.laser.n_state.real, self.laser.n_state.imag],
            },
            "sample_n_medium": self.sample_n_medium,
            "layers": [_layer_to_dict(L) for L in self.layers],
            "extra": self.extra,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimParams":
        laser = d["laser"]
        m = laser["m_state"]
        n = laser["n_state"]
        return cls(
            run=RunParams(**d["run"]),
            laser=LaserParams(
                source_type=laser["source_type"],
                wavelength=laser["wavelength"],
                sigma=laser["sigma"],
                m_state=complex(m[0], m[1]),
                n_state=complex(n[0], n[1]),
            ),
            sample_n_medium=d["sample_n_medium"],
            layers=[_layer_from_dict(L) for L in d["layers"]],
            extra=d.get("extra", {}),
        )

    @classmethod
    def from_json(cls, s: str) -> "SimParams":
        return cls.from_dict(json.loads(s))


# ══════════════════════════════════════════════════════════════════════════════
#  Sensor results
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SensorMeta:
    """Common metadata shared by all sensors (plus the full raw dict)."""
    type: str
    id: str
    hits: int
    absorb_photons: bool
    estimator_enabled: bool
    origin: np.ndarray
    normal: np.ndarray
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, m: Dict[str, Any]) -> "SensorMeta":
        return cls(
            type=str(_py(m.get("type", ""))),
            id=str(_py(m.get("id", ""))),
            hits=int(_py(m.get("hits", 0))),
            absorb_photons=bool(_py(m.get("absorb_photons", False))),
            estimator_enabled=bool(_py(m.get("estimator_enabled", False))),
            origin=np.asarray(m.get("origin", np.zeros(3))),
            normal=np.asarray(m.get("normal", np.zeros(3))),
            raw={k: _py(v) for k, v in m.items()},
        )


@dataclass
class FarFieldCBSResult:
    """Raw far-field CBS Stokes grids (coherent + incoherent), shape (N_t, N_theta, N_phi)."""
    meta: SensorMeta
    S0_coh: np.ndarray
    S1_coh: np.ndarray
    S2_coh: np.ndarray
    S3_coh: np.ndarray
    S0_incoh: np.ndarray
    S1_incoh: np.ndarray
    S2_incoh: np.ndarray
    S3_incoh: np.ndarray
    N_theta: int
    N_phi: int
    N_t: int
    theta_max: float
    phi_max: float

    @property
    def theta(self) -> np.ndarray:
        return np.linspace(0.0, self.theta_max, self.N_theta)

    @property
    def phi(self) -> np.ndarray:
        return np.linspace(0.0, self.phi_max, self.N_phi)

    @classmethod
    def from_h5(cls, meta: Dict[str, Any], data: Dict[str, np.ndarray]) -> "FarFieldCBSResult":
        return cls(
            meta=SensorMeta.from_raw(meta),
            S0_coh=data["S0_coh"], S1_coh=data["S1_coh"], S2_coh=data["S2_coh"], S3_coh=data["S3_coh"],
            S0_incoh=data["S0_incoh"], S1_incoh=data["S1_incoh"],
            S2_incoh=data["S2_incoh"], S3_incoh=data["S3_incoh"],
            N_theta=int(_py(meta.get("N_theta", 0))),
            N_phi=int(_py(meta.get("N_phi", 0))),
            N_t=int(_py(meta.get("N_t", 0))),
            theta_max=float(_py(meta.get("theta_max", 0.0))),
            phi_max=float(_py(meta.get("phi_max", 0.0))),
        )


@dataclass
class PlanarFluenceResult:
    """Raw planar fluence Stokes grids, each shape (N_t, N_x, N_y)."""
    meta: SensorMeta
    S0: np.ndarray
    S1: np.ndarray
    S2: np.ndarray
    S3: np.ndarray
    N_x: int
    N_y: int
    N_t: int
    dx: float
    dy: float
    dt: float

    @classmethod
    def from_h5(cls, meta: Dict[str, Any], data: Dict[str, np.ndarray]) -> "PlanarFluenceResult":
        return cls(
            meta=SensorMeta.from_raw(meta),
            S0=data["S0"], S1=data["S1"], S2=data["S2"], S3=data["S3"],
            N_x=int(_py(meta.get("N_x", 0))),
            N_y=int(_py(meta.get("N_y", 0))),
            N_t=int(_py(meta.get("N_t", 0))),
            dx=float(_py(meta.get("dx", 0.0))),
            dy=float(_py(meta.get("dy", 0.0))),
            dt=float(_py(meta.get("dt", 0.0))),
        )


@dataclass
class PlanarFieldResult:
    """Raw complex planar electric field, shape (N_x, N_y)."""
    meta: SensorMeta
    Ex: np.ndarray
    Ey: np.ndarray
    N_x: int
    N_y: int
    dx: float
    dy: float

    @classmethod
    def from_h5(cls, meta: Dict[str, Any], data: Dict[str, np.ndarray]) -> "PlanarFieldResult":
        return cls(
            meta=SensorMeta.from_raw(meta),
            Ex=data["Ex"], Ey=data["Ey"],
            N_x=int(_py(meta.get("N_x", 0))),
            N_y=int(_py(meta.get("N_y", 0))),
            dx=float(_py(meta.get("dx", 0.0))),
            dy=float(_py(meta.get("dy", 0.0))),
        )


@dataclass
class StatisticsResult:
    """Histograms of detected-photon properties (missing histograms are ``None``)."""
    meta: SensorMeta
    events_histogram: Optional[np.ndarray]
    theta_histogram: Optional[np.ndarray]
    phi_histogram: Optional[np.ndarray]
    depth_histogram: Optional[np.ndarray]
    time_histogram: Optional[np.ndarray]
    weight_histogram: Optional[np.ndarray]

    @classmethod
    def from_h5(cls, meta: Dict[str, Any], data: Dict[str, np.ndarray]) -> "StatisticsResult":
        g = data.get
        return cls(
            meta=SensorMeta.from_raw(meta),
            events_histogram=g("events_histogram"),
            theta_histogram=g("theta_histogram"),
            phi_histogram=g("phi_histogram"),
            depth_histogram=g("depth_histogram"),
            time_histogram=g("time_histogram"),
            weight_histogram=g("weight_histogram"),
        )


@dataclass
class PhotonRecordsResult:
    """Decomposed per-photon records (parallel arrays of length ``n_records``)."""
    meta: SensorMeta
    n_records: int
    events: np.ndarray
    weight: np.ndarray
    penetration_depth: np.ndarray
    launch_time: np.ndarray
    arrival_time: np.ndarray
    r_first: np.ndarray
    r_last: np.ndarray
    E_forward: np.ndarray
    E_reverse: np.ndarray

    @classmethod
    def from_h5(cls, meta: Dict[str, Any], records: Dict[str, np.ndarray]) -> "PhotonRecordsResult":
        empty = np.empty(0)
        g = lambda k: records.get(k, empty)
        return cls(
            meta=SensorMeta.from_raw(meta),
            n_records=int(_py(meta.get("n_records", 0))),
            events=g("events"), weight=g("weight"),
            penetration_depth=g("penetration_depth"),
            launch_time=g("launch_time"), arrival_time=g("arrival_time"),
            r_first=g("r_first"), r_last=g("r_last"),
            E_forward=g("E_forward"), E_reverse=g("E_reverse"),
        )


@dataclass
class AbsorptionResult:
    """
    Cylindrical absorption recorder (time-resolved or time-integrated).

    ``slices`` stacks the raw per-time-bin grids to shape ``(n_slices, n_r, n_z)``
    and ``images`` stacks the corresponding display images.  Index 0 is the
    time-integrated grid; further indices are the time bins (when ``d_t > 0``).
    """
    radius: float
    depth: float
    d_r: float
    d_z: float
    d_t: float
    t_max: float
    n_t: int
    slices: np.ndarray
    images: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """Time-integrated absorption grid (slice 0)."""
        return self.slices[0]

    @property
    def total_image(self) -> np.ndarray:
        """Time-integrated absorption display image (slice 0)."""
        return self.images[0]

    @classmethod
    def from_h5(
        cls,
        meta: Dict[str, Any],
        slices: List[np.ndarray],
        images: List[np.ndarray],
    ) -> "AbsorptionResult":
        return cls(
            radius=float(_py(meta.get("radius", 0.0))),
            depth=float(_py(meta.get("depth", 0.0))),
            d_r=float(_py(meta.get("d_r", 0.0))),
            d_z=float(_py(meta.get("d_z", 0.0))),
            d_t=float(_py(meta.get("d_t", 0.0))),
            t_max=float(_py(meta.get("t_max", 0.0))),
            n_t=int(_py(meta.get("n_t", 0))),
            slices=np.stack(slices, axis=0) if slices else np.empty(0),
            images=np.stack(images, axis=0) if images else np.empty(0),
        )


# Map sensor type name -> builder taking (meta_dict, data_dict).
RESULT_BUILDERS: Dict[str, Callable[[Dict[str, Any], Dict[str, np.ndarray]], Any]] = {
    "FarFieldCBSSensor": FarFieldCBSResult.from_h5,
    "PlanarFluenceSensor": PlanarFluenceResult.from_h5,
    "PlanarFieldSensor": PlanarFieldResult.from_h5,
    "StatisticsSensor": StatisticsResult.from_h5,
    "PhotonRecordSensor": PhotonRecordsResult.from_h5,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Post-processed results
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FarFieldCBSProcessedResult:
    """Normalised far-field CBS, coherent/incoherent Stokes stacked over time bins."""
    theta: np.ndarray
    phi: np.ndarray
    coh_s0: np.ndarray
    coh_s1: np.ndarray
    coh_s2: np.ndarray
    coh_s3: np.ndarray
    inc_s0: np.ndarray
    inc_s1: np.ndarray
    inc_s2: np.ndarray
    inc_s3: np.ndarray

    @property
    def N_t(self) -> int:
        return self.coh_s0.shape[0]

    @property
    def N_theta(self) -> int:
        return self.coh_s0.shape[1]

    @property
    def N_phi(self) -> int:
        return self.coh_s0.shape[2]


@dataclass
class PlanarFluenceProcessedResult:
    """Normalised planar fluence Stokes, each shape (N_t, N_x, N_y)."""
    S0: np.ndarray
    S1: np.ndarray
    S2: np.ndarray
    S3: np.ndarray


@dataclass
class PlanarFieldProcessedResult:
    """Normalised complex planar field."""
    Ex: np.ndarray
    Ey: np.ndarray


# ══════════════════════════════════════════════════════════════════════════════
#  PhotonRecordSensor encoder (registered on the schema)
# ══════════════════════════════════════════════════════════════════════════════

def encode_photon_records(sensor, g, g_meta, g_data, write_dataset, write_attr) -> None:
    """
    Serialise a ``PhotonRecordSensor`` into ``<group>/records/`` parallel arrays.

    *write_dataset* and *write_attr* are the low-level HDF5 writers supplied by
    :mod:`luminis_mc.manager` (passed in to avoid an import cycle).
    """
    rec = sensor.recorded_photons
    n = len(rec)
    write_attr(g_meta, "n_records", n)
    if n == 0:
        return

    rg = g.require_group("records")
    write_dataset(rg, "events",            np.array([r.events for r in rec], dtype=np.int32))
    write_dataset(rg, "weight",            np.array([r.weight for r in rec], dtype=np.float64))
    write_dataset(rg, "penetration_depth", np.array([r.penetration_depth for r in rec], dtype=np.float64))
    write_dataset(rg, "launch_time",       np.array([r.launch_time for r in rec], dtype=np.float64))
    write_dataset(rg, "arrival_time",      np.array([r.arrival_time for r in rec], dtype=np.float64))
    write_dataset(rg, "r_first", np.array(
        [[r.position_first_scattering.x, r.position_first_scattering.y, r.position_first_scattering.z]
         for r in rec], dtype=np.float64))
    write_dataset(rg, "r_last", np.array(
        [[r.position_last_scattering.x, r.position_last_scattering.y, r.position_last_scattering.z]
         for r in rec], dtype=np.float64))
    write_dataset(rg, "E_forward", np.array(
        [[r.polarization_forward.m, r.polarization_forward.n] for r in rec], dtype=np.complex128))
    write_dataset(rg, "E_reverse", np.array(
        [[r.polarization_reverse.m, r.polarization_reverse.n] for r in rec], dtype=np.complex128))


# Register the encoder on the (frozen) schema entry.
schema.SENSOR_SCHEMAS["PhotonRecordSensor"] = replace(
    schema.SENSOR_SCHEMAS["PhotonRecordSensor"], encoder=encode_photon_records
)
