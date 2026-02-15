#pragma once
#include <algorithm>
#include <functional>
#include <vector>
#include <complex>


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

class DataTable {
public:
  std::vector<double> x_values;
  std::vector<std::complex<double>> y_values;

  DataTable() = default;
  void initialize(const std::vector<double>& x_vals, const std::vector<std::complex<double>>& y_vals);
  std::complex<double> Sample(double x) const;
};

} // namespace luminis::sample
