#include "luminis/log/logger.hpp"
#include <luminis/sample/table.hpp>

namespace luminis::sample {

void SamplingTable::initialize(PDFFunction pdfFunc, int nDiv, double minVal, double maxVal) {
  LLOG_DEBUG("Creating SamplingTable with nDiv: {}, minVal: {}, maxVal: {}", nDiv, minVal, maxVal);

  if (nDiv <= 0) {
    LLOG_ERROR("nDiv must be greater than 0, got {}", nDiv);
    throw std::invalid_argument("nDiv must be greater than 0");
  }

  if (minVal >= maxVal) {
    LLOG_ERROR("minVal must be less than maxVal, got minVal: {}, maxVal: {}", minVal, maxVal);
    throw std::invalid_argument("minVal must be less than maxVal");
  }

  double step = (maxVal - minVal) / nDiv;
  values.resize(nDiv + 1);
  cdf.resize(nDiv + 1);

  std::vector<double> pdf(nDiv);
  double sum = 0.0;

  // Compute PDF values and sum for normalization
  for (int i = 0; i < nDiv + 1; i++) {
    values[i] = minVal + i * step;
    pdf[i] = pdfFunc(values[i]);
    sum += pdf[i] * step;
  }

  // Normalize PDF
  for (int i = 0; i < nDiv + 1; i++) {
    pdf[i] /= sum;
  }

  // Compute CDF
  cdf[0] = pdf[0] * step;
  for (int i = 1; i < nDiv + 1; i++) {
    cdf[i] = cdf[i - 1] + pdf[i] * step;
  }

  // Normalize CDF
  for (int i = 0; i < nDiv + 1; i++) {
    cdf[i] /= cdf[nDiv];
  }
}

double SamplingTable::Sample(double u) const {
  if (values.empty() || cdf.empty()) {
    LLOG_ERROR("SamplingTable::Sample called on an uninitialized table");
    throw std::runtime_error("SamplingTable is not initialized");
  }

  auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
  int index = std::distance(cdf.begin(), it);

  if (index == 0)
    return values[0];
  if (index >= values.size())
    return values.back();

  double t = (u - cdf[index - 1]) / (cdf[index] - cdf[index - 1]);
  return values[index - 1] + t * (values[index] - values[index - 1]);
}

void DataTable::initialize(const std::vector<double>& x_vals, const std::vector<std::complex<double>>& y_vals) {
  if (x_vals.size() != y_vals.size()) {
    LLOG_ERROR("DataTable::initialize: x_vals and y_vals must have the same size");
    throw std::invalid_argument("x_vals and y_vals must have the same size");
  }
  x_values = x_vals;
  y_values = y_vals;
}

std::complex<double> DataTable::Sample(double x) const {
  if (x_values.empty() || y_values.empty()) {
    LLOG_ERROR("DataTable::Sample called on an uninitialized table");
    throw std::runtime_error("DataTable is not initialized");
  }

  auto it = std::lower_bound(x_values.begin(), x_values.end(), x);
  int index = std::distance(x_values.begin(), it);

  if (index == 0)
    return y_values[0];
  if (index >= x_values.size())
    return y_values.back();

  double t = (x - x_values[index - 1]) / (x_values[index] - x_values[index - 1]);
  return y_values[index - 1] + t * (y_values[index] - y_values[index - 1]);
}

} // namespace luminis::sample
