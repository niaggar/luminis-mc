#include <cmath>
#include <complex>
#include <luminis/core/detector.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/log/logger.hpp>
#include <vector>

namespace luminis::core {

Detector::Detector(double z) {
  origin = {0, 0, z};
  normal = Z_UNIT_VEC3;
  backward_normal = Z_UNIT_VEC3 * -1;
  m_polarization = X_UNIT_VEC3;
  n_polarization = Y_UNIT_VEC3;
}

Detector Detector::copy_start() const {
  Detector det(origin.z);
  return det;
}

void Detector::merge_from(const Detector &other) {
  hits += other.hits;
  recorded_photons.insert(recorded_photons.end(),
                          other.recorded_photons.begin(),
                          other.recorded_photons.end());
}

bool Detector::is_hit_by(const Photon &photon) const {
  const Vec3 xn = photon.prev_pos;
  const Vec3 xf = photon.pos;
  const Vec3 d{
    xf.x - xn.x,
    xf.y - xn.y,
    xf.z - xn.z
  };

  const double denom = dot(d, normal);

  // Plane and line are parallel
  if (std::abs(denom) < 1e-6)
    return false;

  const double t = dot({
    origin.x - xn.x,
    origin.y - xn.y,
    origin.z - xn.z
  }, normal) / denom;

  // Intersection point is not between xn and xf
  if (t < 0 || t > 1)
    return false;

  return true;
}

void Detector::record_hit(Photon &photon) {
  const Vec3 xn = photon.prev_pos;
  const Vec3 xf = photon.pos;
  const Vec3 d{
    xf.x - xn.x,
    xf.y - xn.y,
    xf.z - xn.z
  };

  const double denom = dot(d, normal);

  // Plane and line are parallel
  if (std::abs(denom) < 1e-6)
    return;

  const double t = dot({
    origin.x - xn.x,
    origin.y - xn.y,
    origin.z - xn.z
  }, normal) / denom;

  // Intersection point is not between xn and xf
  if (t < 0 || t > 1)
    return;

  const Vec3 hit_point{
    xn.x + t * d.x,
    xn.y + t * d.y,
    xn.z + t * d.z
  };
  const double correction_distance = luminis::math::norm({
    hit_point.x - xf.x,
    hit_point.y - xf.y,
    hit_point.z - xf.z
  });
  if (correction_distance > 0) {
    photon.opticalpath -= correction_distance;
  }

  hits += 1;
  photon.pos = hit_point;

  PhotonRecord photon_rec{};
  photon_rec.events = photon.events;
  photon_rec.penetration_depth = photon.penetration_depth;
  photon_rec.launch_time = photon.launch_time;
  photon_rec.arrival_time = photon.launch_time + (photon.opticalpath / photon.velocity);
  photon_rec.opticalpath = photon.opticalpath;
  photon_rec.weight = photon.weight;
  photon_rec.position_detector = photon.pos;
  photon_rec.position_first_scattering = photon.r_0;
  photon_rec.position_last_scattering = photon.r_n;
  photon_rec.direction = photon.dir;
  photon_rec.m = photon.m;
  photon_rec.n = photon.n;
  photon_rec.polarization_forward = photon.polarization;
  photon_rec.polarization_reverse = CVec2{std::complex<double>(0, 0), std::complex<double>(0, 0)};

  recorded_photons.push_back(photon_rec);
}

} // namespace luminis::core
