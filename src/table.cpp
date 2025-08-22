#include "luminis/log/logger.hpp"
#include <luminis/sample/table.hpp>

namespace luminis::sample {

SamplingTable::SamplingTable(PDFFunction pdfFunc, int nDiv, double minVal, double maxVal) {
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
  values.resize(nDiv);
  cdf.resize(nDiv);

  std::vector<double> pdf(nDiv);
  double sum = 0.0;

  // Compute PDF values and sum for normalization
  for (int i = 0; i < nDiv; i++) {
    values[i] = minVal + i * step;
    pdf[i] = pdfFunc(values[i]);
    sum += pdf[i] * step;
  }

  // Normalize PDF
  for (int i = 0; i < nDiv; i++) {
    pdf[i] /= sum;
  }

  // Compute CDF
  cdf[0] = pdf[0] * step;
  for (int i = 1; i < nDiv; i++) {
    cdf[i] = cdf[i - 1] + pdf[i] * step;
  }

  // Normalize CDF
  for (int i = 0; i < nDiv; i++) {
    cdf[i] /= cdf[nDiv - 1];
  }
}

double SamplingTable::Sample(double u) {
  auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
  int index = std::distance(cdf.begin(), it);

  if (index == 0)
    return values[0];
  if (index >= values.size())
    return values.back();

  double t = (u - cdf[index - 1]) / (cdf[index] - cdf[index - 1]);
  return values[index - 1] + t * (values[index] - values[index - 1]);
}

} // namespace luminis::sample
