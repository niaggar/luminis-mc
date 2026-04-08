/// @file absortion.hpp
/// @brief Unified absorption recorder for Monte Carlo light transport simulations.
///
/// Records deposited weight on a cylindrical (r, z) grid, optionally resolved
/// in time.
///
/// Time resolution follows the same convention as PlanarFluenceSensor:
///   - `d_t == 0` → time-integrated (single time bin)
///   - `d_t > 0`  → `n_t = ceil(t_max / d_t) + 1`
///                   - bin 0: always time-integrated
///                   - bins 1..n_t-1: explicit time windows
///
/// Thread safety: Absorption is NOT thread-safe. For parallel simulations,
/// each thread works on a cloned copy (via clone()). After all threads finish,
/// results are combined via merge_from().

#pragma once
#include <cstddef>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <vector>
#include <optional>

using namespace luminis::math;

namespace luminis::core {

/// @brief Absorption recorder on a cylindrical (r, z) grid with optional time resolution.
///
/// Each time bin stores an `n_r × n_z` Matrix of accumulated deposited weight.
/// When `d_t == 0`, a single time bin is used (time-integrated mode).
/// When `d_t > 0`, photon arrival time is computed from `launch_time + opticalpath / velocity`
/// and deposited both in bin 0 (integrated) and in one explicit time-window bin.
struct Absorption {
  double radius;   ///< Maximum radial extent of the recording grid.
  double depth;    ///< Maximum depth (z) extent of the recording grid.
  double d_r;      ///< Radial bin width.
  double d_z;      ///< Depth bin width.
  double d_t;      ///< Time bin width (0 for time-integrated).
  double t_max;    ///< Total time window (ignored when d_t == 0).
  int n_t;         ///< Number of time bins (1 when time-integrated; else ceil(t_max/d_t)+1).

  /// @brief Per-time-bin absorption grids. Each element is an n_r × n_z Matrix.
  /// @details For d_t > 0: index 0 is always integrated, 1..n_t-1 are time windows.
  std::vector<Matrix> time_slices;

  /// @brief Construct an absorption recorder.
  /// @param r      Maximum radial extent.
  /// @param z      Maximum depth extent.
  /// @param dr     Radial bin width (n_r = floor(r/dr) + 1).
  /// @param dz     Depth bin width (n_z = floor(z/dz) + 1).
  /// @param dt     Time bin width. Use 0 for time-integrated recording.
  /// @param t_max  Total time window. Ignored when dt == 0.
  Absorption(double r, double z, double dr, double dz, double dt = 0.0, double t_max = 0.0);

  /// @brief Create an empty clone with identical configuration but zeroed grids.
  Absorption clone() const;

  /// @brief Accumulate another absorption recorder's data into this one (element-wise addition).
  void merge_from(const Absorption &other);

  /// @brief Record deposited weight from a photon at its current position (and time).
  /// @param photon   The photon whose position (and arrival time) determines the bin.
  /// @param d_weight The fraction of weight deposited by implicit absorption.
  void record_absorption(const Photon &photon, double d_weight);

  /// @brief Produce a symmetric 2D absorption image for a given time bin.
  /// @param n_photons   Total number of simulated photons (for normalization).
  /// @param time_index  Index of the time bin to visualize (default 0).
  /// @return An (2·n_r) × n_z Matrix normalized by voxel volume and photon count,
  ///         mirrored along the radial axis.
  Matrix get_absorption_image(int n_photons, int time_index = 0) const;

  /// @brief Produce a symmetric 2D absorption image from the integrated slice.
  /// @details This is equivalent to get_absorption_image(n_photons, 0).
  /// @param n_photons   Total number of simulated photons (for normalization).
  /// @return An (2·n_r) × n_z Matrix normalized by voxel volume and photon count,
  ///         mirrored along the radial axis.
  Matrix get_total_image(int n_photons) const;
};

} // namespace luminis::core
