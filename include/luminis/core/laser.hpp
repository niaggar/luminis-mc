#pragma once
#include <luminis/core/photon.hpp>
#include <luminis/math/rng.hpp>
#include <luminis/math/vec.hpp>

using namespace luminis::math;

namespace luminis::core {

enum class LaserSource {
  Point = 0,
  Uniform = 1,
  Gaussian = 2,
};

enum class TemporalProfile {
  Delta,
  Gaussian,
  TopHat,
  Exponential,
  PulseTrain,
  CW
};

struct Laser {
  Vec3 position;
  Vec3 direction;
  Vec3 local_m;
  Vec3 local_n;
  Vec2 polarization;

  double wavelength;
  double sigma;
  LaserSource source_type;

  TemporalProfile temporal_profile{TemporalProfile::Delta};
  double pulse_duration{0.0}; // in ps
  double repetition_rate{0.0}; // in Hz
  double time_offset{0.0}; // in ps


  Laser(Vec3 position, Vec3 direction, Vec3 local_m, Vec3 local_n,
        Vec2 polarization, double wavelength, double sigma,
        LaserSource source_type);

  void set_temporal_profile(TemporalProfile profile, double pulse_duration = 0.0, double repetition_rate = 0.0, double time_offset = 0.0) {
    temporal_profile = profile;
    this->pulse_duration = pulse_duration;
    this->repetition_rate = repetition_rate;
    this->time_offset = time_offset;
  }
  double sample_emission_time(Rng &rng) const;
  Photon emit_photon(Rng &rng) const;
};

Vec3 uniform_distribution(Rng &rng, const Vec3 &center, const double sigma);
Vec3 gaussian_distribution(Rng &rng, const Vec3 &center, const double sigma);

} // namespace luminis::core
