#include <luminis/sample/phase.hpp>
#include <luminis/log/logger.hpp>
#include <cmath>
#include <vector>

namespace luminis::sample {

double form_factor(const double theta, const double k, const double radius)
{
  const double ks = 2.0 * k * std::sin(theta / 2.0);
  const double numerator = 3 * (std::sin(ks * radius) - ks * radius * std::cos(ks * radius));
  const double denominator = std::pow(ks * radius, 3);

  return numerator / denominator;
}



double PhaseFunction::sample_phi(double x) {
  return 2.0 * M_PI * x; // Uniformly sample phi in [0, 2pi)
}
double PhaseFunction::sample_phi_conditional(double theta, CVec2& S, CVec2& E, double k, Rng& rng) {
    const double s2sq = std::norm(S[0]); // |S2|^2
    const double s1sq = std::norm(S[1]); // |S1|^2

    const double e1sq = std::norm(E[0]);
    const double e2sq = std::norm(E[1]);
    const double e12  = std::real(E[0] * std::conj(E[1]));

    const double a = s2sq*e1sq + s1sq*e2sq;
    const double b = s1sq*e1sq + s2sq*e2sq;
    const double c = 2.0*(s2sq - s1sq)*e12;
    const double diff = (a - b);
    const double Fmax = 0.5*(a + b) + 0.5*std::sqrt(diff*diff + c*c);

    while (true) {
      const double phi = 2.0*M_PI*rng.uniform();
      const double cp = std::cos(phi), sp = std::sin(phi);
      const double F  = a*cp*cp + b*sp*sp + c*cp*sp;
      // protección numérica mínima
      const double Fclamped = std::max(F, 0.0);
      if (rng.uniform()*Fmax <= Fclamped) return phi;
    }
}
std::array<double, 2> PhaseFunction::get_anisotropy_factor(Rng& rng, std::size_t nSamples) {
  double sum = 0.0;
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < nSamples; ++i) {
    const double mu = rng.uniform();
    const double c  = this->sample_cos(mu);
    sum += c;
    sum_sq += c * c;
  }

  const double g = sum / static_cast<double>(nSamples);
  const double variance = (sum_sq / static_cast<double>(nSamples)) - (g * g);
  const double stddev = std::sqrt(variance / static_cast<double>(nSamples));
  return {g, stddev};
}



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
  const double F = form_factor(x, k, radius);
  const double a = 1.0 / (4.0 * M_PI);
  return a * std::pow(F, 2);
}



RayleighDebyeEMCPhaseFunction::RayleighDebyeEMCPhaseFunction(double wavelength, double radius, int nDiv, double minVal, double maxVal) {
  if (radius <= 0.0) {
    LLOG_WARN("RayleighDebyeEMCPhaseFunction: radius must be positive, got {}", radius);
  }
  if (wavelength <= 0.0) {
    LLOG_WARN("RayleighDebyeEMCPhaseFunction: wavelength must be positive, got {}", wavelength);
  }

  this->radius = radius;
  this->wavelength = wavelength;
  this->k = 2 * M_PI / wavelength;
  this->table.initialize([this](double x) { return this->PDF(x); }, nDiv, minVal, maxVal);
}
double RayleighDebyeEMCPhaseFunction::sample_cos(double x) {
  return cos(table.Sample(x));
}
double RayleighDebyeEMCPhaseFunction::sample_theta(double x) {
  return table.Sample(x); // Return theta
}
double RayleighDebyeEMCPhaseFunction::PDF(double x) {
  const double F = form_factor(x, k, radius);
  const double a = 1.0 / (4.0 * M_PI);
  return a * std::pow(F, 2) * (1.0 + std::pow(cos(x), 2));
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
