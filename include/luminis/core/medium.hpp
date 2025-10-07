#pragma once
#include "luminis/core/absortion.hpp"
#include <luminis/math/rng.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/sample/phase.hpp>

using namespace luminis::math;
using namespace luminis::sample;

namespace luminis::core {

struct Medium {
  AbsorptionTimeDependent *absorption{nullptr};
  PhaseFunction *phase_function{nullptr};

  double mu_absorption{0.0}; // Absorption coefficient [1/mm]
  double mu_scattering{0.0}; // Scattering coefficient [1/mm]
  double mu_attenuation{0.0}; // Attenuation coefficient [1/mm]

  const double light_speed{299792458e-6}; // Speed of light in medium [mm/ns]
  double refractive_index{1.0};     // Refractive index of the medium

  virtual ~Medium() = default;
  Medium(double absorption, double scattering, PhaseFunction *phase_func);

  virtual double sample_free_path(Rng &rng) const = 0;
  virtual double sample_azimuthal_angle(Rng &rng) const;
  virtual double sample_scattering_angle(Rng &rng) const = 0;
  virtual CVec2 scattering_matrix(const double theta, const double phi, const double k) const = 0;
  double light_speed_in_medium() const;
};

struct SimpleMedium : public Medium {
  double mean_free_path; // Mean free path [mm]
  double radius;         // Radius of the particles [mm]

  SimpleMedium(double absorption, double scattering, PhaseFunction *phase_func, double mfp, double r);

  double sample_free_path(Rng &rng) const override;
  double sample_scattering_angle(Rng &rng) const override;
  CVec2 scattering_matrix(const double theta, const double phi,
                          const double k) const override;
};

} // namespace luminis::core
