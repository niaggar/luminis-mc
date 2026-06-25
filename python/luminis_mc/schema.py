"""
schema.py
=========
Declarative registry describing how each sensor type is persisted.

This is the **single source of truth** used by both the writing side
(:py:meth:`luminis_mc.manager.Experiment.save_sensor`) and the reading side
(:py:meth:`luminis_mc.manager.ResultsLoader`).  Adding support for a new
sensor type means adding one :class:`SensorSchema` entry here — no logic has to
change anywhere else.

Each sensor is stored under ``/sensors/<name>/`` as two sub-groups:

- ``meta/`` — the common base attributes (:data:`COMMON_META`) plus the
  type-specific scalar attributes listed in ``meta_attrs`` (HDF5 attributes).
- ``data/`` — the array-valued attributes listed in ``data_attrs`` (HDF5
  datasets).  Empty arrays are skipped and flagged with a ``skipped:<attr>``
  meta attribute.

The ``PhotonRecordSensor`` is special: its per-photon records are decomposed
into several parallel arrays under a ``records/`` sub-group by a dedicated
``encoder``/``decoder`` pair (see :mod:`luminis_mc.records`).
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple


# ── Attributes shared by every sensor type (base ``Sensor``) ──────────────────
COMMON_META: Tuple[str, ...] = (
    "id",
    "origin",
    "normal",
    "backward_normal",
    "n_polarization",
    "m_polarization",
    "hits",
    "absorb_photons",
    "estimator_enabled",
    "filter_theta_enabled",
    "filter_theta_min",
    "filter_theta_max",
    "filter_phi_enabled",
    "filter_phi_min",
    "filter_phi_max",
    "filter_position_enabled",
    "filter_x_min",
    "filter_x_max",
    "filter_y_min",
    "filter_y_max",
    "filter_direction_enabled",
    "filter_direction",
)


@dataclass(frozen=True)
class SensorSchema:
    """
    Description of how a single sensor type maps to the HDF5 file.

    Parameters
    ----------
    type_name:
        Sensor class name (``sensor.__class__.__name__``), used as the registry
        key and stored as ``meta/type``.
    meta_attrs:
        Names of scalar attributes (beyond :data:`COMMON_META`) written to
        ``meta/`` as HDF5 attributes.
    data_attrs:
        Names of array-valued attributes written to ``data/`` as HDF5 datasets.
    encoder:
        Optional callable ``(group, sensor, g_meta, g_data) -> None`` that fully
        owns serialization for irregular sensors (e.g. ``PhotonRecordSensor``).
        When set, ``data_attrs`` is ignored on write.
    """

    type_name: str
    meta_attrs: Tuple[str, ...] = ()
    data_attrs: Tuple[str, ...] = ()
    encoder: Optional[Callable] = field(default=None, compare=False)


# ── Registry ──────────────────────────────────────────────────────────────────
# NOTE: the attribute lists mirror the C++ bindings (see python/luminis_mc/_core.pyi)
# and were lifted 1:1 from the previous hand-written save_sensor() if/elif block.

SENSOR_SCHEMAS = {
    "FarFieldCBSSensor": SensorSchema(
        "FarFieldCBSSensor",
        meta_attrs=(
            "N_theta", "N_phi", "N_t", "t_max",
            "theta_max", "phi_max", "dtheta", "dphi", "theta_pp_max",
        ),
        data_attrs=(
            "S0_coh", "S1_coh", "S2_coh", "S3_coh",
            "S0_incoh", "S1_incoh", "S2_incoh", "S3_incoh",
        ),
    ),
    "PlanarFluenceSensor": SensorSchema(
        "PlanarFluenceSensor",
        meta_attrs=("N_x", "N_y", "N_t", "dx", "dy", "dt", "len_x", "len_y", "len_t"),
        data_attrs=("S0", "S1", "S2", "S3"),
    ),
    "PlanarFieldSensor": SensorSchema(
        "PlanarFieldSensor",
        meta_attrs=("N_x", "N_y", "dx", "dy", "len_x", "len_y"),
        data_attrs=("Ex", "Ey"),
    ),
    "StatisticsSensor": SensorSchema(
        "StatisticsSensor",
        meta_attrs=(
            "events_histogram_bins_set", "max_events",
            "theta_histogram_bins_set", "min_theta", "max_theta", "n_bins_theta", "dtheta",
            "phi_histogram_bins_set", "min_phi", "max_phi", "n_bins_phi", "dphi",
            "depth_histogram_bins_set", "max_depth", "n_bins_depth", "ddepth",
            "N_t", "dt", "t_max",
            "time_histogram_bins_set", "h_max_time", "n_bins_time", "h_dtime",
            "weight_histogram_bins_set", "max_weight", "n_bins_weight", "dweight",
        ),
        data_attrs=(
            "events_histogram", "theta_histogram", "phi_histogram",
            "depth_histogram", "time_histogram", "weight_histogram",
        ),
    ),
    # PhotonRecordSensor uses a custom encoder (set in records.py to avoid an
    # import cycle); meta_attrs/data_attrs are filled in by the encoder.
    "PhotonRecordSensor": SensorSchema(
        "PhotonRecordSensor",
        meta_attrs=(),
        data_attrs=(),
    ),
}


def get_schema(type_name: str) -> SensorSchema:
    """Return the :class:`SensorSchema` for *type_name* or raise ``KeyError``."""
    try:
        return SENSOR_SCHEMAS[type_name]
    except KeyError:
        raise KeyError(
            f"Unsupported sensor type: '{type_name}'. "
            f"Add a SensorSchema entry in luminis_mc/schema.py. "
            f"Known types: {sorted(SENSOR_SCHEMAS)}"
        )
