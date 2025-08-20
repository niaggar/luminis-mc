#pragma once
#include <luminis/sample/table.hpp>

namespace luminis::sample {

class PhaseFunction {
public:
  virtual ~PhaseFunction() = default;
  virtual double Sample(double x) = 0;
};

class UniformPhaseFunction : public PhaseFunction {
public:
  double Sample(double x) override;
};

class RayleighPhaseFunction : public PhaseFunction {
public:
  RayleighPhaseFunction(int nDiv, double minVal, double maxVal);
  double Sample(double x) override;
  double PDF(double x);
private:
  SamplingTable table;
};

class HenyeyGreensteinPhaseFunction : public PhaseFunction {
public:
  HenyeyGreensteinPhaseFunction(double g);
  double Sample(double x) override;
private:
  double g; // Anisotropy factor
};

} // namespace luminis::sample
