#include <luminis/core/detector.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/log/logger.hpp>

namespace luminis::core {

Detector::Detector(const Vec3 o, const Vec3 normal, const Vec3 n, const Vec3 m)
    : origin(o), normal(normal), n_polarization(n), m_polarization(m) {}

void Detector::record_hit(Photon &photon) {
  const Vec3 xn = photon.prev_pos;
  const Vec3 xf = photon.pos;
  const Vec3 d = xf - xn;

  const double denom = dot(d, normal);

  // Plane and line are parallel
  if (std::abs(denom) < 1e-6)
    return;

  const double t = dot(origin - xn, normal) / denom;

  // Intersection point is not between xn and xf
  if (t < 0 || t > 1)
    return;

  const Vec3 hit_point = xn + d * t;
  const double correction_distance = norm(hit_point - photon.prev_pos);
  if (correction_distance > 0) {
    photon.opticalpath -= correction_distance;
  }

  // LLOG_DEBUG("Photon hit detector at position: {}", hit_point);

  hits += 1;
  photon.alive = false;
  photon.pos = hit_point;
  recorded_photons.push_back(std::move(photon));
}

} // namespace luminis::core
