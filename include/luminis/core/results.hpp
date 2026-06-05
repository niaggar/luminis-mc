/**
 * @file results.hpp
 * @brief Post-processing of raw sensor accumulators into normalized results.
 *
 * Sensors accumulate raw, unnormalized data during transport (summed Stokes
 * parameters, summed fields, raw coherent/incoherent intensities). The helpers
 * here convert those raw accumulators into physical quantities by dividing out
 * the photon count, pixel area, and/or solid angle, returning plain "Processed"
 * structs that are convenient to expose to Python.
 *
 * @see detector.hpp — the sensors whose accumulators are post-processed here.
 */

#pragma once
#include <cstddef>
#include <luminis/core/detector.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/math/utils.hpp>
#include <vector>
#include <functional>
#include <memory>
#include <map>

namespace luminis::core
{
  /// @brief Stokes parameters (S0..S3) on a 2D grid.
  struct StokesMatrixProcessed
  {
    Matrix S0, S1, S2, S3;
  };

  /// @brief Stokes parameters (S0..S3) as radial profiles (one value per radius/angle bin).
  struct StokesRadialProcessed
  {
    std::vector<double> S0, S1, S2, S3;
  };

  /// @brief Normalized far-field CBS result: coherent and incoherent Stokes per time bin.
  struct FarFieldCBSProcessed
  {
    Matrix dOmega;                                    ///< Solid angle per (theta, phi) bin [sr]. [N_theta x N_phi]
    std::vector<StokesMatrixProcessed> coherent;      ///< Coherent (interfering) Stokes grids, one per time bin.
    std::vector<StokesMatrixProcessed> incoherent;    ///< Incoherent (baseline) Stokes grids, one per time bin.
  };

  /// @brief Normalized planar fluence result: Stokes grids, one per time bin. [N_x x N_y]
  struct PlanarFluenceProcessed
  {
    std::vector<Matrix> S0, S1, S2, S3;
  };

  /// @brief Normalized planar field result: complex electric field components. [N_x x N_y]
  struct PlanarFieldProcessed
  {
    CMatrix Ex, Ey;
  };

  /// @brief Azimuthally-averaged far-field CBS result (radial profile of Stokes parameters).
  struct FarFieldCBSRadialProcessed
  {
    std::vector<double> theta_center;     ///< Center angle of each radial (theta) bin [rad].
    StokesRadialProcessed coherent;       ///< Coherent radial Stokes profile.
    StokesRadialProcessed incoherent;     ///< Incoherent radial Stokes profile.
  };

  /// @brief Normalize a PlanarFluenceSensor's accumulated Stokes grids.
  /// @param det                  Source sensor holding the raw accumulators.
  /// @param n_photons            Total simulated photons (per-photon normalization).
  /// @param normalize_per_photon Divide by photon count.
  /// @param normalize_per_area   Divide by pixel area.
  /// @param eps                  Threshold guarding divisions by near-zero area.
  PlanarFluenceProcessed postprocess_planar_fluence(const PlanarFluenceSensor &det, std::size_t n_photons, bool normalize_per_photon = true, bool normalize_per_area = true, double eps = 1e-30);

  /// @brief Normalize a PlanarFieldSensor's accumulated complex field components.
  /// @see postprocess_planar_fluence for the shared normalization parameters.
  PlanarFieldProcessed postprocess_planar_field(const PlanarFieldSensor &det, std::size_t n_photons, bool normalize_per_photon = true, bool normalize_per_area = true, double eps = 1e-30);

  /// @brief Normalize a FarFieldCBSSensor's coherent/incoherent Stokes grids by solid angle and photon count.
  /// @param det                        Source CBS sensor holding the raw accumulators.
  /// @param n_photons                  Total simulated photons (per-photon normalization).
  /// @param normalize_per_solid_angle  Divide each bin by its solid angle.
  /// @param normalize_per_photon       Divide by photon count.
  /// @param eps                        Threshold guarding divisions by near-zero solid angle.
  FarFieldCBSProcessed postprocess_farfield_cbs(const FarFieldCBSSensor &det, std::size_t n_photons, bool normalize_per_solid_angle = true, bool normalize_per_photon = true, double eps = 1e-30);
  // FarFieldCBSRadialProcessed radial_average_S0(const FarFieldCBSSensor &det, std::size_t n_photons, bool normalize_per_solid_angle = true, bool normalize_per_photon = true, double eps = 1e-30);
}
