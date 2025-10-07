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
  virtual double sample_phi(double x);
  virtual double sample_phi_conditional(double theta, CVec2& S, CVec2& E, double k, Rng& rng);
  virtual double sample_cos(double x) = 0;
  virtual double sample_theta(double x) = 0;
  double get_g();
};

class UniformPhaseFunction : public PhaseFunction {
public:
  double sample_cos(double x) override;
  double sample_theta(double x) override;
};

class RayleighPhaseFunction : public PhaseFunction {
public:
  RayleighPhaseFunction(int nDiv, double minVal, double maxVal);
  double sample_cos(double x) override;
  double sample_theta(double x) override;
  double PDF(double x);
private:
  SamplingTable table;
};

class HenyeyGreensteinPhaseFunction : public PhaseFunction {
public:
  HenyeyGreensteinPhaseFunction(double g);
  double sample_cos(double x) override;
  double sample_theta(double x) override;
private:
  double g; // Anisotropy factor
};

class RayleighDebyePhaseFunction : public PhaseFunction {
public:
  RayleighDebyePhaseFunction(double wavelenght, double radius, int nDiv, double minVal, double maxVal);
  double sample_cos(double x) override;
  double sample_theta(double x) override;
  double PDF(double x);
private:
  SamplingTable table;
  double wavelength;
  double radius;
  double k;
};

class RayleighDebyeEMCPhaseFunction : public PhaseFunction {
public:
  RayleighDebyeEMCPhaseFunction(double wavelenght, double radius, int nDiv, double minVal, double maxVal);
  double sample_cos(double x) override;
  double sample_theta(double x) override;
  double PDF(double x);
private:
  SamplingTable table;
  double wavelength;
  double radius;
  double k;
};

class DrainePhaseFunction : public PhaseFunction {
public:
  DrainePhaseFunction(double g, double a, int nDiv, double minVal, double maxVal);
  double sample_cos(double x) override;
  double sample_theta(double x) override;
  double PDF(double x);
private:
  SamplingTable table;
  double g; // Anisotropy factor
  double a; // Draine parameter
};

} // namespace luminis::sample
