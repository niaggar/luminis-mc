/**
 * @file photon.cpp
 * @brief Implementation of the Photon packet construction and utility methods.
 *
 * This file implements the three methods declared in photon.hpp:
 *
 * - `Photon::Photon(...)` — initializes position, local frame, and wave number.
 * - `Photon::set_polarization(...)` — overrides the Jones vector.
 * - `Photon::get_stokes_parameters()` — computes Stokes parameters from the
 *   current Jones vector using the standard I, Q, U, V definitions.
 *
 * @see photon.hpp for full field documentation and design notes.
 */

#include "luminis/math/vec.hpp"
#include <luminis/core/photon.hpp>

namespace luminis::core
{
  // ═══════════════════════════════════════════════════════════════════════════
  // Photon implementation
  // ═══════════════════════════════════════════════════════════════════════════

  Photon::Photon(const Vec3 &p, const Vec3 &d, const Vec3 &m, const Vec3 &n, const double wl)
  {
    // --- Position initialization ---
    // Both pos and prev_pos are set to p so the first record_hit call sees
    // zero displacement and skips all z-plane intersection tests.
    this->pos.x = p.x;
    this->pos.y = p.y;
    this->pos.z = p.z;

    this->prev_pos.x = p.x;
    this->prev_pos.y = p.y;
    this->prev_pos.z = p.z;

    // --- Wavelength and wave number ---
    this->wavelength_nm = wl;
    this->k = 2.0 * M_PI / wavelength_nm;

    // --- Local frame (P_local) ---
    // Rows of P_local store the right-handed orthonormal basis (m, n, s=d):
    //   Row 0: m  (transverse "horizontal")
    //   Row 1: n  (transverse "vertical")
    //   Row 2: d  (propagation direction)
    P_local(0, 0) = m.x;
    P_local(0, 1) = m.y;
    P_local(0, 2) = m.z;
    P_local(1, 0) = n.x;
    P_local(1, 1) = n.y;
    P_local(1, 2) = n.z;
    P_local(2, 0) = d.x;
    P_local(2, 1) = d.y;
    P_local(2, 2) = d.z;
  }

  void Photon::set_polarization(CVec2 pol)
  {
    // Mark the photon as polarized and overwrite the Jones vector.
    // The local frame (P_local) is unchanged; the caller is responsible
    // for ensuring pol is expressed in the current frame.
    this->polarized = true;
    this->polarization.m = pol.m;
    this->polarization.n = pol.n;
  }

  std::array<double, 4> Photon::get_stokes_parameters() const
  {
    // Return unpolarized Stokes vector for incoherent transport.
    if (!polarized)
    {
      return {1.0, 0.0, 0.0, 0.0};
    }

    // Jones components: (Em, En) = (polarization.m, polarization.n)
    const auto &Ex = polarization.m;
    const auto &Ey = polarization.n;

    // Standard Stokes-Jones relations:
    //   S0 (I) = |Ex|² + |Ey|²    — total intensity
    //   S1 (Q) = |Ex|² − |Ey|²    — linear polarization along m / n axes
    //   S2 (U) = 2 Re(Ex·Ey*)     — linear polarization at ±45°
    //   S3 (V) = −2 Im(Ex·Ey*)    — circular polarization (RHC positive)
    const double I = std::norm(Ex) + std::norm(Ey);
    const double Q = std::norm(Ex) - std::norm(Ey);
    const double U = 2.0 * std::real(Ex * std::conj(Ey));
    const double V = -2.0 * std::imag(Ex * std::conj(Ey));

    return {I, Q, U, V};
  }

} // namespace luminis::core
