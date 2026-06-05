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
class DataTable {
public:
  std::vector<double> x_values;                ///< Strictly increasing abscissae.
  std::vector<std::complex<double>> y_values;  ///< Function values at each abscissa.

  DataTable() = default;

  /// @brief Populate the table from precomputed (x, y) pairs.
  void initialize(const std::vector<double>& x_vals, const std::vector<std::complex<double>>& y_vals);

  /// @brief Return the interpolated value y(x). @param x Query abscissa.
  std::complex<double> Sample(double x) const;
};

} // namespace luminis::sample
