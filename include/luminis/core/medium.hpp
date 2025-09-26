#pragma once
#include <luminis/math/rng.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/sample/phase.hpp>

using namespace luminis::math;
using namespace luminis::sample;

namespace luminis::core {

struct Medium {
  PhaseFunction* phase_function{nullptr};

  double mu_a{0.0}; // Absorption coefficient [1/mm]
  double mu_s{0.0}; // Scattering coefficient [1/mm]

  Medium(double absorption, double scattering, PhaseFunction* phase_func);

  virtual double sample_free_path(Rng &rng) const;
  virtual double sample_scattering_angle(Rng &rng) const;
  virtual double sample_azimuthal_angle(Rng &rng) const;
};


struct SimpleMedium : public Medium
{
  double mean_free_path; // Mean free path [mm]

  SimpleMedium(double absorption, double scattering, PhaseFunction* phase_func, double mfp);

  double sample_free_path(Rng &rng) const override;
  double sample_scattering_angle(Rng &rng) const override;
  double sample_azimuthal_angle(Rng &rng) const override;
};

} // namespace luminis::core
