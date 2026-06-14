/**
 * @file table.hpp
 * @brief Lookup tables for inverse-CDF sampling and tabulated complex functions.
 *
 * - **SamplingTable**: builds a cumulative distribution from an arbitrary PDF and
 *   draws samples by inverse-CDF lookup (used by the phase functions).
 * - **DataTable**: stores a tabulated complex-valued function y(x) and returns
 *   interpolated values (used to cache the Mie amplitudes S1/S2).
 */

#pragma once
#include <algorithm>
#include <functional>
#include <vector>
#include <complex>


namespace luminis::sample {

/// @brief Callable returning the (unnormalized) probability density at a point.
using PDFFunction = std::function<double(double)>;

/// @brief Inverse-CDF sampling table built from a probability density function.
class SamplingTable {
public:
  std::vector<double> values; ///< Sampled abscissae (the grid over [minVal, maxVal]).
  std::vector<double> cdf;    ///< Normalized cumulative distribution at each abscissa.

  SamplingTable() = default;

  /// @brief Build the table by integrating `pdfFunc` over a uniform grid.
  /// @param pdfFunc Probability density function to sample from.
  /// @param nDiv    Number of grid divisions.
  /// @param minVal  Lower bound of the sampling domain.
  /// @param maxVal  Upper bound of the sampling domain.
  void initialize(PDFFunction pdfFunc, int nDiv, double minVal, double maxVal);

  /// @brief Draw a sample via inverse-CDF lookup.
  /// @param u Uniform random value in [0, 1).
  /// @return Sampled value interpolated from the table.
  double Sample(double u) const ;
};

/// @brief Tabulated complex function y(x) with interpolated lookup.
///
/// Two layouts are supported:
///   - **Arbitrary grid** (`initialize`): abscissae may be non-uniform; lookup uses a
///     binary search (`std::lower_bound`) followed by linear interpolation.
///   - **Uniform grid** (`initialize_uniform`): abscissae are equally spaced; lookup is
///     O(1) via direct index computation `idx = (x - x_min) * inv_dx`. This is the hot
///     path for the per-bin scattering-matrix lookups in the CBS estimator.
class DataTable {
public:
  std::vector<double> x_values;                ///< Strictly increasing abscissae (also filled for uniform grids, for compatibility).
  std::vector<std::complex<double>> y_values;  ///< Function values at each abscissa.

  bool uniform_{false};   ///< True when the grid is uniform (enables O(1) direct indexing).
  double x_min_{0.0};     ///< First abscissa (uniform grid only).
  double dx_{0.0};        ///< Grid spacing (uniform grid only).
  double inv_dx_{0.0};    ///< Reciprocal grid spacing 1/dx (uniform grid only).

  DataTable() = default;

  /// @brief Populate the table from precomputed (x, y) pairs on an arbitrary grid.
  void initialize(const std::vector<double>& x_vals, const std::vector<std::complex<double>>& y_vals);

  /// @brief Populate the table from values on a uniform grid x_i = x_min + i·dx.
  /// @param x_min First abscissa.
  /// @param dx    Grid spacing (> 0).
  /// @param y_vals Function values at each node (size N; covers [x_min, x_min+(N-1)·dx]).
  /// @details Enables the O(1) direct-index lookup path in Sample().
  void initialize_uniform(double x_min, double dx, const std::vector<std::complex<double>>& y_vals);

  /// @brief Return the interpolated value y(x). @param x Query abscissa.
  std::complex<double> Sample(double x) const;
};

} // namespace luminis::sample
