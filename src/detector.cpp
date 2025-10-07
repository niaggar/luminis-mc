#include <cmath>
#include <complex>
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

std::vector<double> Detector::compute_events_histogram(const double min_theta, const double max_theta) {
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

AngularSpeckle Detector::compute_speckle_maps(const int n_theta, const int n_phi) {
  std::vector<std::vector<CVec2>> Ebin(n_theta, std::vector<CVec2>(n_phi));
  std::vector<std::vector<std::complex<double>>> Enormal(n_theta, std::vector<std::complex<double>>(n_phi));

  const Vec3 backward_normal = -1 * normal;

  for (auto &ph : recorded_photons) {
      // 1) calcula (theta, phi) como arriba → (it, ip)
      Vec3 u = normalize(ph.dir);
      const double cos_theta = dot(u, backward_normal) / (norm(u) * norm(backward_normal));
      const double theta = std::acos(cos_theta);           // [0, pi]

      Vec3 u_par = u - dot(u, normal)*normal;        // componente en el plano
      double un = dot(u_par, n_polarization);
      double um = dot(u_par, m_polarization);
      double phi = std::atan2(um, un);               // (-pi, pi]
      if (phi < 0) phi += 2.0*M_PI;                  // [0, 2pi)

      int it = std::min(int(theta / (0.5*M_PI) * n_theta), n_theta-1);
      int ip = std::min(int(phi / (2.0*M_PI) * n_phi), n_phi-1);



      // 2) fase de propagación:
      std::complex<double> phase = std::exp(std::complex<double>(0, ph.k * ph.opticalpath));
      std::complex<double> En_local_photon = ph.polarization[0] * phase * std::sqrt(ph.weight);
      std::complex<double> Em_local_photon = ph.polarization[1] * phase * std::sqrt(ph.weight);

      // 3) suma coherente en la celda (it, ip)
      const Vec3 n_pho = ph.n;
      const Vec3 m_pho = ph.m;
      const Vec3 n_det = n_polarization;
      const Vec3 m_det = m_polarization;
      const std::complex<double> En = En_local_photon * dot(n_det, n_pho) + Em_local_photon * dot(n_det, m_pho);
      const std::complex<double> Em = En_local_photon * dot(m_det, n_pho) + Em_local_photon * dot(m_det, m_pho);
      const std::complex<double> Enorm = En * dot(backward_normal, n_det) + Em * dot(backward_normal, m_det);

      Ebin[it][ip][0] += En;
      Ebin[it][ip][1] += Em;
      Enormal[it][ip] += Enorm;
  }

  std::vector<std::vector<double>> Ix(n_theta, std::vector<double>(n_phi, 0.0));
  std::vector<std::vector<double>> Iy(n_theta, std::vector<double>(n_phi, 0.0));
  std::vector<std::vector<double>> I (n_theta, std::vector<double>(n_phi, 0.0));

  for (int it=0; it<n_theta; ++it) {
    for (int ip=0; ip<n_phi; ++ip) {
      auto &C = Ebin[it][ip];
      Ix[it][ip] = std::norm(C[0]);
      Iy[it][ip] = std::norm(C[1]);
      I [it][ip] = Ix[it][ip] + Iy[it][ip] + std::norm(Enormal[it][ip]);
    }
  }

  AngularSpeckle result;
  result.Ix = std::move(Ix);
  result.Iy = std::move(Iy);
  result.I  = std::move(I);
  result.N_theta = n_theta;
  result.N_phi   = n_phi;
  return result;
}

} // namespace luminis::core
