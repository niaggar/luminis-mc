#pragma once
#include <luminis/math/rng.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/sample/phase.hpp>

using namespace luminis::math;
using namespace luminis::sample;

namespace luminis::core {

double form_factor(const double theta, const double k, const double radius);

struct Medium {
  PhaseFunction* phase_function{nullptr};

  double mu_a{0.0}; // Absorption coefficient [1/mm]
  double mu_s{0.0}; // Scattering coefficient [1/mm]

  Medium(double absorption, double scattering, PhaseFunction* phase_func);

  virtual double sample_free_path(Rng &rng) const = 0;
  virtual double sample_scattering_angle(Rng &rng) const = 0;
  virtual double sample_azimuthal_angle(Rng &rng) const = 0;
  virtual CVec2 scattering_matrix(const double theta, const double phi, const double k) const = 0;
};


struct SimpleMedium : public Medium
{
  double mean_free_path; // Mean free path [mm]
  double radius;         // Radius of the particles [mm]

  SimpleMedium(double absorption, double scattering, PhaseFunction* phase_func, double mfp, double r);

  double sample_free_path(Rng &rng) const override;
  double sample_scattering_angle(Rng &rng) const override;
  double sample_azimuthal_angle(Rng &rng) const override;
  CVec2 scattering_matrix(const double theta, const double phi, const double k) const override;
};

} // namespace luminis::core
