"""
analysis.py
===========
Composable post-processing of CBS results.

The recurring analysis pattern across the figure scripts is always the same
pipeline applied to a :class:`~luminis_mc.FarFieldCBSProcessedResult`::

    pick a time window  ->  decompose into polarization channels  ->  reduce phi
                        ->  (optionally) coherent / incoherent enhancement

Only the *pieces* change: sometimes you keep the 2D (theta, phi) map, sometimes
you azimuthally average; the channel decomposition can be circular (uses S0, S3)
or linear (uses S0, S1).  Instead of re-defining a ``Profiles`` class and a
``load_profiles`` helper in every script, build the profiles with
:func:`cbs_profiles` and swap the ``basis`` / ``reduce`` pieces:

    from results.utils.analysis import cbs_profiles, circular, linear, keep

    prof = cbs_profiles(proc, basis=circular, time_index=10)   # 1D, helicity
    ax.plot(prof.theta * 1e3, prof.coherent["total"])
    ax.plot(prof.theta * 1e3, prof.enhancement["co"])

    prof_lin = cbs_profiles(proc, basis=linear)                # linear channels
    prof_2d  = cbs_profiles(proc, reduce=keep)                 # keep (theta, phi)

A *basis* is any ``Callable[[Stokes], dict[str, np.ndarray]]`` and a *reducer*
any ``Callable[[np.ndarray], np.ndarray]`` — so you can pass your own inline
without touching this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from luminis_mc import FarFieldCBSProcessedResult


# ── Stokes bundle ─────────────────────────────────────────────────────────────

@dataclass
class Stokes:
    """The four Stokes components at a chosen time window (any shared shape)."""
    S0: np.ndarray
    S1: np.ndarray
    S2: np.ndarray
    S3: np.ndarray


Basis = Callable[[Stokes], Dict[str, np.ndarray]]
Reducer = Callable[[np.ndarray], np.ndarray]


# ── Bases: Stokes -> named polarization channels ─────────────────────────────

def circular(s: Stokes) -> Dict[str, np.ndarray]:
    """Circular (helicity) channels: ``co`` preserved, ``cross`` reversed."""
    co = (s.S0 - s.S3) / 2.0
    cross = (s.S0 + s.S3) / 2.0
    return {"co": co, "cross": cross, "total": s.S0}


def linear(s: Stokes) -> Dict[str, np.ndarray]:
    """Linear channels: ``co`` parallel, ``cross`` perpendicular."""
    co = (s.S0 + s.S1) / 2.0
    cross = (s.S0 - s.S1) / 2.0
    return {"co": co, "cross": cross, "total": s.S0}


def total(s: Stokes) -> Dict[str, np.ndarray]:
    """Only the total intensity channel (S0)."""
    return {"total": s.S0}


def raw_stokes(s: Stokes) -> Dict[str, np.ndarray]:
    """The raw Stokes components as channels."""
    return {"S0": s.S0, "S1": s.S1, "S2": s.S2, "S3": s.S3}


# ── Reducers: collapse the phi axis (last axis) ──────────────────────────────

def azimuthal_average(a: np.ndarray) -> np.ndarray:
    """Average over phi (last axis) -> 1D theta profile. 1D input passes through."""
    a = np.asarray(a)
    return a.mean(axis=-1) if a.ndim >= 2 else a


def keep(a: np.ndarray) -> np.ndarray:
    """No reduction: keep the full (theta, phi) map."""
    return np.asarray(a)


def phi_cut(index: int = 0) -> Reducer:
    """Return a reducer that takes a single phi column (a phi cut)."""
    return lambda a: np.asarray(a)[..., index]


# ── Profiles container ───────────────────────────────────────────────────────

@dataclass
class CBSProfiles:
    """
    Coherent / incoherent intensity per channel, plus lazy enhancement.

    ``coherent`` and ``incoherent`` map channel name -> array (1D theta profile,
    or 2D map if ``reduce=keep``).  ``enhancement`` is computed on demand as the
    coherent/incoherent ratio per channel.
    """
    theta: np.ndarray
    coherent: Dict[str, np.ndarray]
    incoherent: Dict[str, np.ndarray]
    eps: float = 1e-30

    @property
    def enhancement(self) -> Dict[str, np.ndarray]:
        return {
            k: (self.coherent[k] + self.eps) / (self.incoherent[k] + self.eps)
            for k in self.coherent
        }

    def channels(self) -> List[str]:
        return list(self.coherent)


def _select_time(stack: np.ndarray, time_index: Optional[int]) -> np.ndarray:
    """Pick a time bin (``time_index``) or sum over all bins (``None``)."""
    stack = np.asarray(stack)
    if time_index is None:
        return stack.sum(axis=0)
    return stack[time_index]


def cbs_profiles(
    proc: FarFieldCBSProcessedResult,
    *,
    basis: Basis = circular,
    reduce: Reducer = azimuthal_average,
    time_index: Optional[int] = 0,
    eps: float = 1e-30,
) -> CBSProfiles:
    """
    Build :class:`CBSProfiles` from a processed CBS result.

    Parameters
    ----------
    proc:
        A :class:`~luminis_mc.FarFieldCBSProcessedResult`
        (``loader.processed_cbs(...)``).
    basis:
        Channel decomposition (:func:`circular`, :func:`linear`, :func:`total`,
        :func:`raw_stokes`, or any ``Stokes -> dict`` callable).
    reduce:
        Phi handling (:func:`azimuthal_average`, :func:`keep`, :func:`phi_cut`,
        or any ``ndarray -> ndarray`` callable).
    time_index:
        Time window to analyze; ``None`` sums over all time bins.
    eps:
        Regularizer for the enhancement ratio.
    """
    coh = Stokes(
        _select_time(proc.coh_s0, time_index),
        _select_time(proc.coh_s1, time_index),
        _select_time(proc.coh_s2, time_index),
        _select_time(proc.coh_s3, time_index),
    )
    inc = Stokes(
        _select_time(proc.inc_s0, time_index),
        _select_time(proc.inc_s1, time_index),
        _select_time(proc.inc_s2, time_index),
        _select_time(proc.inc_s3, time_index),
    )
    coherent = {k: reduce(v) for k, v in basis(coh).items()}
    incoherent = {k: reduce(v) for k, v in basis(inc).items()}
    return CBSProfiles(theta=proc.theta, coherent=coherent, incoherent=incoherent, eps=eps)
