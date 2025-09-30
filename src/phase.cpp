#include <luminis/sample/phase.hpp>
#include <luminis/log/logger.hpp>
#include <cmath>

namespace luminis::sample {

double UniformPhaseFunction::sample_cos(double x) {
  return 2.0*x - 1.0; // Return cos(theta)
}
double UniformPhaseFunction::sample_theta(double x) {
  return acos(sample_cos(x));
}

RayleighPhaseFunction::RayleighPhaseFunction(int nDiv, double minVal, double maxVal) {
  this->table.initialize([this](double x) { return this->PDF(x); }, nDiv, minVal, maxVal);
}
double RayleighPhaseFunction::sample_cos(double x) {
  return table.Sample(x); // Return cos(theta)
}
double RayleighPhaseFunction::sample_theta(double x) {
  return acos(sample_cos(x));
}
double RayleighPhaseFunction::PDF(double x) {
  return 0.375 * (1.0 + x * x);
}

HenyeyGreensteinPhaseFunction::HenyeyGreensteinPhaseFunction(double g) {
  if (g < -1.0 || g > 1.0) {
    LLOG_WARN("HenyeyGreensteinPhaseFunction: g parameter must be between -1 and 1, got {}", g);
  }
  this->g = g;
}
double HenyeyGreensteinPhaseFunction::sample_cos(double x) {
  const double gg = g * g;
	const double frac = (1.0 - gg) / (1.0 - g + 2.0 * g * x);
	return (1.0 + gg - frac * frac) / (2.0 * g); // Return cos(theta)
}
double HenyeyGreensteinPhaseFunction::sample_theta(double x) {
  return acos(sample_cos(x));
}

RayleighDebyePhaseFunction::RayleighDebyePhaseFunction(double wavelength, double radius, int nDiv, double minVal, double maxVal) {
  if (radius <= 0.0) {
    LLOG_WARN("RayleighDebyePhaseFunction: radius must be positive, got {}", radius);
  }
  if (wavelength <= 0.0) {
    LLOG_WARN("RayleighDebyePhaseFunction: wavelength must be positive, got {}", wavelength);
  }

  this->radius = radius;
  this->wavelength = wavelength;
  this->k = 2 * M_PI / wavelength;
  this->table.initialize([this](double x) { return this->PDF(x); }, nDiv, minVal, maxVal);
}
double RayleighDebyePhaseFunction::sample_cos(double x) {
  return cos(table.Sample(x));
}
double RayleighDebyePhaseFunction::sample_theta(double x) {
  return table.Sample(x); // Return theta
}
double RayleighDebyePhaseFunction::PDF(double x) {
 	const double ks = 2 * k * sin(x/2);
	const double numerator = 3 * (sin(ks * radius) - ks * radius * cos(ks * radius));
	const double denominator = pow(ks * radius, 3);
	if (denominator == 0) {
    LLOG_WARN("RayleighDebyePhaseFunction::PDF: denominator is zero, returning 0.0");
    return 0.0;
  }
	return pow(k, 2) / (4 * M_PI) * pow(numerator / denominator, 2);
}


DrainePhaseFunction::DrainePhaseFunction(double g, double a, int nDiv, double minVal, double maxVal) {
  if (g < -1.0 || g > 1.0) {
    LLOG_WARN("DrainePhaseFunction: g parameter must be between -1 and 1, got {}", g);
  }
  if (a < -1.0 || a > 1.0) {
    LLOG_WARN("DrainePhaseFunction: a parameter must be between -1 and 1, got {}", a);
  }

  this->g = g;
  this->a = a;
  this->table.initialize([this](double x) { return this->PDF(x); }, nDiv, minVal, maxVal);
}
double DrainePhaseFunction::sample_cos(double x) {
  return table.Sample(x); // Return cos(theta)
}
double DrainePhaseFunction::sample_theta(double x) {
  return acos(sample_cos(x));
}
double DrainePhaseFunction::PDF(double x) {
  const double hgTerm = (1.0 - g*g) / pow(1.0+g*g-2.0*g*x, 1.5);
	const double draineTerm = (1.0 + a*x*x) / (3 + a*(1+2*g*g));
	return (1.0 / 4.0 * M_PI) * hgTerm * draineTerm;
}

}
