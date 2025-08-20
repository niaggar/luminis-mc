#include <luminis/sample/table.hpp>

namespace luminis::sample {

SamplingTable::SamplingTable(PDFFunction pdfFunc, int nDiv, double minVal, double maxVal) {
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
