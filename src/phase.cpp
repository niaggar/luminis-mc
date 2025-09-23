#include <luminis/sample/phase.hpp>
#include <luminis/log/logger.hpp>
#include <cmath>

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

RayleighDebyePhaseFunction::RayleighDebyePhaseFunction(double wavelength, double radius, int nDiv, double minVal, double maxVal)
    : table([this](double x) { return this->PDF(x); }, nDiv, minVal, maxVal), wavelength(wavelength), radius(radius) {
  k = 2.0 * M_PI / wavelength;
}
double RayleighDebyePhaseFunction::Sample(double x) {
  return table.Sample(x);
}
double RayleighDebyePhaseFunction::PDF(double x) {
 	double ks = 2 * k * sin(x/2);
	double numerator = 3 * (sin(ks * radius) - ks * radius * cos(ks * radius));
	double denominator = pow(ks * radius, 3);
	if (denominator == 0) {
    LLOG_WARN("RayleighDebyePhaseFunction::PDF: denominator is zero, returning 0.0");
    return 0.0;
  }
	return pow(k, 2) / (4 * M_PI) * pow(numerator / denominator, 2);
}

}
