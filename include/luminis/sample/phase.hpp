#pragma once
#include <luminis/sample/table.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/math/rng.hpp>
#include <complex>

using namespace luminis::math;

namespace luminis::sample {

double form_factor(const double theta, const double k, const double radius);

class PhaseFunction {
public:
  virtual ~PhaseFunction() = default;
  virtual double sample_phi(double x) const;
  virtual double sample_phi_conditional(double theta, CMatrix& S, CVec2& E, double k, Rng& rng) const;
  virtual double sample_cos(double x) const = 0;
  virtual double sample_theta(double x) const = 0;
  std::array<double, 2> get_anisotropy_factor(Rng& rng, std::size_t n_samples = 200000) const;
};

class UniformPhaseFunction : public PhaseFunction {
public:
  double sample_cos(double x) const override;
  double sample_theta(double x) const override;
};

class RayleighPhaseFunction : public PhaseFunction {
public:
  RayleighPhaseFunction(int nDiv, double minVal, double maxVal);
  double sample_cos(double x) const override;
  double sample_theta(double x) const override;
  double PDF(double x);
private:
  SamplingTable table;
};

class HenyeyGreensteinPhaseFunction : public PhaseFunction {
public:
  HenyeyGreensteinPhaseFunction(double g);
  double sample_cos(double x) const override;
  double sample_theta(double x) const override;
private:
  double g; // Anisotropy factor
};

class RayleighDebyePhaseFunction : public PhaseFunction {
public:
  RayleighDebyePhaseFunction(double wavelenght, double radius, int nDiv, double minVal, double maxVal);
  double sample_cos(double x) const override;
  double sample_theta(double x) const override;
  double PDF(double x);
private:
  SamplingTable table;
  double wavelength;
  double radius;
  double k;
};

class RayleighDebyeEMCPhaseFunction : public PhaseFunction {
public:
  RayleighDebyeEMCPhaseFunction(double wavelenght, double radius, double n_particle, double n_medium, int nDiv, double minVal, double maxVal);
  double sample_cos(double x) const override;
  double sample_theta(double x) const override;
  double PDF(double x);
private:
  SamplingTable table;
  double wavelength;
  double radius;
  double k;
  double n_particle;
  double n_medium;
};

class DrainePhaseFunction : public PhaseFunction {
public:
  DrainePhaseFunction(double g, double a, int nDiv, double minVal, double maxVal);
  double sample_cos(double x) const override;
  double sample_theta(double x) const override;
  double PDF(double x);
private:
  SamplingTable table;
  double g; // Anisotropy factor
  double a; // Draine parameter
};

class MiePhaseFunction : public PhaseFunction {
public:
  MiePhaseFunction(double wavelength, double radius, double n_particle, double n_medium, int nDiv, double minVal, double maxVal);
  double sample_cos(double x) const override;
  double sample_theta(double x) const override;
  double PDF(double x);
private:
  SamplingTable table;
  double wavelength;
  double radius;
  double k;
  double n_particle;
  double n_medium;
  std::complex<double> m; // Relative refractive index
};

} // namespace luminis::sample
