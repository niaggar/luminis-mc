#include <luminis/sample/phase.hpp>

namespace luminis::sample {

double UniformPhaseFunction::Sample(double x) {
  return 2.0*x - 1.0;
}

RayleighPhaseFunction::RayleighPhaseFunction(int nDiv, double minVal, double maxVal)
    : table([this](double x) { return this->PDF(x); }, nDiv, minVal, maxVal) {}
double RayleighPhaseFunction::Sample(double x) {
  return table.Sample(x);
}
double RayleighPhaseFunction::PDF(double x) {
  return 0.375 * (1.0 + x * x);
}

HenyeyGreensteinPhaseFunction::HenyeyGreensteinPhaseFunction(double g) : g(g) {}
double HenyeyGreensteinPhaseFunction::Sample(double x) {
  double gg = g * g;
	double frac = (1.0 - gg) / (1.0 - g + 2.0 * g * x);
	return (1.0 + gg - frac * frac) / (2.0 * g);
}

}
