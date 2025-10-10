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

  // LLOG_DEBUG("Photon hit detector at position: {}", hit_point);

  hits += 1;
  photon.alive = false;
  photon.pos = hit_point;
  recorded_photons.push_back(std::move(photon));
}

std::vector<double> Detector::compute_events_histogram(const double min_theta, const double max_theta) {
  const Vec3 backward_normal{
      -1 * normal.x,
      -1 * normal.y,
      -1 * normal.z
  };
  int max_hit_number = 0;
  for (const auto &photon : recorded_photons) {
    if (photon.events > max_hit_number) {
      max_hit_number = photon.events;
    }
  }

  const int n_bins = max_hit_number + 1;
  std::vector<double> histogram(n_bins, 0.0);

  for (const auto &photon : recorded_photons) {
    const double cos_theta = dot(photon.dir, backward_normal) / (luminis::math::norm(photon.dir) * luminis::math::norm(backward_normal));
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

AngularIntensity Detector::compute_speckle(const int n_theta, const int n_phi) {
  std::vector<std::vector<CVec2>> Ebin(n_theta, std::vector<CVec2>(n_phi));
  std::vector<std::vector<std::complex<double>>> Enormal(n_theta, std::vector<std::complex<double>>(n_phi));

  const Vec3 backward_normal{
    -1 * normal.x,
    -1 * normal.y,
    -1 * normal.z
  };

  for (auto &ph : recorded_photons) {
    // 1) calcula (theta, phi) como arriba → (it, ip)
    Vec3 u = ph.dir;
    const double cos_theta = dot(u, backward_normal) / (luminis::math::norm(u) * luminis::math::norm(backward_normal));
    const double theta = std::acos(cos_theta);           // [0, pi]

    Vec3 u_par{
      u.x - dot(u, normal) * normal.x,
      u.y - dot(u, normal) * normal.y,
      u.z - dot(u, normal) * normal.z
    };
    double un = dot(u_par, n_polarization);
    double um = dot(u_par, m_polarization);
    double phi = std::atan2(um, un);               // (-pi, pi]
    if (phi < 0) phi += 2.0*M_PI;                  // [0, 2pi)

    int it = std::min(int(theta / (0.5*M_PI) * n_theta), n_theta-1);
    int ip = std::min(int(phi / (2.0*M_PI) * n_phi), n_phi-1);

    // 2) fase de propagación:
    std::complex<double> phase = std::exp(std::complex<double>(0, ph.k * ph.opticalpath));
    std::complex<double> Em_local_photon = ph.polarization.m * phase * std::sqrt(ph.weight);
    std::complex<double> En_local_photon = ph.polarization.n * phase * std::sqrt(ph.weight);

    // 3) suma coherente en la celda (it, ip)
    const Vec3 m_pho = ph.m;
    const Vec3 n_pho = ph.n;
    const Vec3 m_det = m_polarization;
    const Vec3 n_det = n_polarization;
    const std::complex<double> En = En_local_photon * dot(n_det, n_pho) + Em_local_photon * dot(n_det, m_pho);
    const std::complex<double> Em = En_local_photon * dot(m_det, n_pho) + Em_local_photon * dot(m_det, m_pho);
    const std::complex<double> Enorm = En_local_photon * dot(backward_normal, n_det) + Em_local_photon * dot(backward_normal, m_det);

    Ebin[it][ip].m += Em;
    Ebin[it][ip].n += En;
    Enormal[it][ip] += Enorm;
  }

  std::vector<std::vector<double>> Ix(n_theta, std::vector<double>(n_phi, 0.0));
  std::vector<std::vector<double>> Iy(n_theta, std::vector<double>(n_phi, 0.0));
  std::vector<std::vector<double>> I (n_theta, std::vector<double>(n_phi, 0.0));

  for (int it=0; it<n_theta; ++it) {
    for (int ip=0; ip<n_phi; ++ip) {
      auto &C = Ebin[it][ip];
      Ix[it][ip] = std::norm(C.m);
      Iy[it][ip] = std::norm(C.n);
      I [it][ip] = Ix[it][ip] + Iy[it][ip] + std::norm(Enormal[it][ip]);
    }
  }

  AngularIntensity result;
  result.Ix = std::move(Ix);
  result.Iy = std::move(Iy);
  result.I  = std::move(I);
  result.N_theta = n_theta;
  result.N_phi   = n_phi;
  return result;
}

SpatialIntensity Detector::compute_spatial_intensity(const double max_theta, const int n_x, const int n_y, const double x_max, const double y_max) {
  SpatialIntensity result(n_x, n_y, x_max, y_max);

  std::vector<std::vector<CVec2>> Ebin(n_x, std::vector<CVec2>(n_y));
  std::vector<std::vector<std::complex<double>>> Enormal(n_x, std::vector<std::complex<double>>(n_y));

  const Vec3 backward_normal{
    -1 * normal.x,
    -1 * normal.y,
    -1 * normal.z
  };

  for (auto &ph : recorded_photons) {
    // Filtra por ángulo de incidencia
    const double cos_theta = dot(ph.dir, backward_normal) / (luminis::math::norm(ph.dir) * luminis::math::norm(backward_normal));
    const double theta = std::acos(cos_theta) * (180.0 / M_PI); // Convert to degrees
    if (theta > max_theta) {
      continue;
    }

    // 1) calcula (x,y) como arriba → (ix, iy)
    Vec3 r{
      ph.pos.x - origin.x,
      ph.pos.y - origin.y,
      ph.pos.z - origin.z
    };
    const double x = dot(r, n_polarization);
    const double y = dot(r, m_polarization);

    int ix = std::min(int((x + x_max) / (2.0 * x_max) * n_x), n_x-1);
    int iy = std::min(int((y + y_max) / (2.0 * y_max) * n_y), n_y-1);

    if (ix < 0 || ix >= n_x || iy < 0 || iy >= n_y) {
      continue;
    }

    // 2) fase de propagación:
    std::complex<double> phase = std::exp(std::complex<double>(0, ph.k * ph.opticalpath));
    std::complex<double> En_local_photon = ph.polarization.n * phase * std::sqrt(ph.weight);
    std::complex<double> Em_local_photon = ph.polarization.m * phase * std::sqrt(ph.weight);

    // 3) suma coherente en la celda (ix, iy)
    const Vec3 n_pho = ph.n;
    const Vec3 m_pho = ph.m;
    const Vec3 n_det = n_polarization;
    const Vec3 m_det = m_polarization;
    const std::complex<double> En = En_local_photon * dot(n_det, n_pho) + Em_local_photon * dot(n_det, m_pho);
    const std::complex<double> Em = En_local_photon * dot(m_det, n_pho) + Em_local_photon * dot(m_det, m_pho);
    const std::complex<double> Enorm = En_local_photon * dot(backward_normal, n_det) + Em_local_photon * dot(backward_normal, m_det);

    Ebin[ix][iy].n += En;
    Ebin[ix][iy].m += Em;
    Enormal[ix][iy] += Enorm;
  }

  for (int ix=0; ix<n_x; ++ix) {
    for (int iy=0; iy<n_y; ++iy) {
      auto &C = Ebin[ix][iy];
      result.Ix[ix][iy] = std::norm(C.m);
      result.Iy[ix][iy] = std::norm(C.n);
      result.I [ix][iy] = result.Ix[ix][iy] + result.Iy[ix][iy] + std::norm(Enormal[ix][iy]);
    }
  }

  return result;
}

AngularIntensity Detector::compute_angular_intensity(const double max_theta, const double max_phi, const int n_theta, const int n_phi) {
  std::vector<std::vector<CVec2>> Ebin(n_theta, std::vector<CVec2>(n_phi));
  std::vector<std::vector<std::complex<double>>> Enormal(n_theta, std::vector<std::complex<double>>(n_phi));

  const Vec3 backward_normal{
    -1 * normal.x,
    -1 * normal.y,
    -1 * normal.z
  };

  for (auto &ph : recorded_photons) {
    // 1) calcula (theta, phi) como arriba → (it, ip)
    Vec3 u = ph.dir;
    const double cos_theta = dot(u, backward_normal) / (luminis::math::norm(u) * luminis::math::norm(backward_normal));
    const double theta = std::acos(cos_theta);           // [0, pi]

    Vec3 u_par{
      u.x - dot(u, normal) * normal.x,
      u.y - dot(u, normal) * normal.y,
      u.z - dot(u, normal) * normal.z
    };
    double un = dot(u_par, n_polarization);
    double um = dot(u_par, m_polarization);
    double phi = std::atan2(um, un);               // (-pi, pi]
    if (phi < 0) phi += 2.0*M_PI;                  // [0, 2pi)

    if (theta > max_theta || phi > max_phi) {
      continue;
    }

    int it = std::min(int(theta / max_theta * n_theta), n_theta-1);
    int ip = std::min(int(phi / max_phi * n_phi), n_phi-1);

    // 2) fase de propagación:
    std::complex<double> phase = std::exp(std::complex<double>(0, ph.k * ph.opticalpath));
    std::complex<double> En_local_photon = ph.polarization.n * phase * std::sqrt(ph.weight);
    std::complex<double> Em_local_photon = ph.polarization.m * phase * std::sqrt(ph.weight);

    // 3) suma coherente en la celda (it, ip)
    const Vec3 n_pho = ph.n;
    const Vec3 m_pho = ph.m;
    const Vec3 n_det = n_polarization;
    const Vec3 m_det = m_polarization;
    const std::complex<double> En = En_local_photon * dot(n_det, n_pho) + Em_local_photon * dot(n_det, m_pho);
    const std::complex<double> Em = En_local_photon * dot(m_det, n_pho) + Em_local_photon * dot(m_det, m_pho);
    const std::complex<double> Enorm = En_local_photon * dot(backward_normal, n_det) + Em_local_photon * dot(backward_normal, m_det);
    Ebin[it][ip].n += En;
    Ebin[it][ip].m += Em;
    Enormal[it][ip] += Enorm;
  }

  std::vector<double> theta_edges(n_theta+1), phi_edges(n_phi+1);
  for (int it=0; it<=n_theta; ++it) theta_edges[it] = (max_theta * it) / n_theta;
  for (int ip=0; ip<=n_phi;   ++ip) phi_edges[ip]   = (max_phi   * ip) / n_phi;

  std::vector<std::vector<double>> Ix(n_theta, std::vector<double>(n_phi, 0.0));
  std::vector<std::vector<double>> Iy(n_theta, std::vector<double>(n_phi, 0.0));
  std::vector<std::vector<double>> I (n_theta, std::vector<double>(n_phi, 0.0));

  for (int it=0; it<n_theta; ++it) {
    const double th0 = theta_edges[it];
    const double th1 = theta_edges[it+1];
    const double dcos = std::cos(th0) - std::cos(th1);

    for (int ip=0; ip<n_phi; ++ip) {
      const double ph0 = phi_edges[ip];
      const double ph1 = phi_edges[ip+1];
      const double dphi = (ph1 - ph0);
      const double domega = dcos * dphi;

      auto &C = Ebin[it][ip];
      Ix[it][ip] = std::norm(C.m) / domega;
      Iy[it][ip] = std::norm(C.n) / domega;
      I [it][ip] = Ix[it][ip] + Iy[it][ip] + (std::norm(Enormal[it][ip]) / domega);
    }
  }

  AngularIntensity result;
  result.Ix = std::move(Ix);
  result.Iy = std::move(Iy);
  result.I  = std::move(I);
  result.N_theta = n_theta;
  result.N_phi   = n_phi;
  result.theta_max = max_theta;
  result.phi_max   = max_phi;

  return result;
}


} // namespace luminis::core
