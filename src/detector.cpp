#include <luminis/core/detector.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/log/logger.hpp>
#include <vector>

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

std::vector<double> Detector::get_hit_histogram(const double min_theta, const double max_theta) {
  const Vec3 backward_normal = -1 * normal;
  int max_hit_number = 0;
  for (const auto &photon : recorded_photons) {
    if (photon.events > max_hit_number) {
      max_hit_number = photon.events;
    }
  }

  const int n_bins = max_hit_number + 1;
  std::vector<double> histogram(n_bins, 0.0);

  for (const auto &photon : recorded_photons) {
    const double cos_theta = dot(photon.dir, backward_normal) / (norm(photon.dir) * norm(backward_normal));
    const double theta = std::acos(cos_theta) * (180.0 / M_PI); // Convert to degrees

    if (theta >= min_theta && theta <= max_theta) {
      if (photon.events < n_bins) {
        histogram[photon.events] += 1.0;
      } else {
        LLOG_WARN("Photon events {} exceed histogram bins {}", photon.events, n_bins);
      }
    }
  }

  return histogram;
}

std::vector<std::vector<double>> Detector::get_hit_angular_distribution() {
  return std::vector<std::vector<double>>{};
}

} // namespace luminis::core
