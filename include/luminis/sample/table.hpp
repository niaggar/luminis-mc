#pragma once
#include <algorithm>
#include <functional>
#include <vector>

namespace luminis::sample {

using PDFFunction = std::function<double(double)>;

class SamplingTable {
public:
  std::vector<double> values;
  std::vector<double> cdf;

  SamplingTable() = default;
  void initialize(PDFFunction pdfFunc, int nDiv, double minVal, double maxVal);

  double Sample(double u) const ;
};

} // namespace luminis::sample
