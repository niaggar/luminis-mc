#include <cmath>
#include <complex>
#include <luminis/core/detector.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/log/logger.hpp>
#include <fstream>
#include <vector>

namespace luminis::core {

Detector::Detector(double z) {
  origin = {0, 0, z};
  normal = {0, 0, 1};
  backward_normal = {0, 0, -1};
  n_polarization = {1, 0, 0};
  m_polarization = {0, 1, 0};
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
  photon.alive = false;
  photon.pos = hit_point;

  PhotonRecord photon_rec{};
  photon_rec.velocity = photon.velocity;
  photon_rec.wavelength_nm = photon.wavelength_nm;
  photon_rec.k = photon.k;
  photon_rec.events = photon.events;
  photon_rec.penetration_depth = photon.penetration_depth;
  photon_rec.launch_time = photon.launch_time;
  photon_rec.arrival_time = photon.launch_time + (photon.opticalpath / photon.velocity);
  photon_rec.opticalpath = photon.opticalpath;
  photon_rec.weight = photon.weight;
  photon_rec.position = photon.pos;
  photon_rec.direction = photon.dir;
  photon_rec.m = photon.m;
  photon_rec.n = photon.n;
  photon_rec.polarization = photon.polarization;

  recorded_photons.push_back(photon_rec);
}

std::vector<double> Detector::compute_events_histogram(const double min_theta, const double max_theta) const {
  int max_hit_number = 0;
  for (const auto &photon : recorded_photons) {
    if (photon.events > max_hit_number) {
      max_hit_number = photon.events;
    }
  }

  const int n_bins = max_hit_number + 1;
  std::vector<double> histogram(n_bins, 0.0);

  for (const auto &photon : recorded_photons) {
    const Vec3 u = photon.direction;
    const double costtheta = -1 * u.z;
    const double theta = std::acos(costtheta);

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

std::vector<double> Detector::compute_theta_histogram(const double min_theta, const double max_theta, const int n_bins) const {
  std::vector<double> histogram(n_bins, 0.0);

  for (const auto &photon : recorded_photons) {
    const Vec3 u = photon.direction;
    const double costtheta = -1 * u.z;
    const double theta = std::acos(costtheta);

    if (theta >= min_theta && theta <= max_theta) {
      const int bin_index = std::min(static_cast<int>(std::floor(((theta - min_theta) / (max_theta - min_theta)) * n_bins)), n_bins - 1);
      histogram[bin_index] += 1.0;
    }
  }

  return histogram;
}

std::vector<double> Detector::compute_phi_histogram(const double min_phi, const double max_phi, const int n_bins) const {
  std::vector<double> histogram(n_bins, 0.0);

  for (const auto &photon : recorded_photons) {
    const Vec3 u = photon.direction;
    double phi = std::atan2(u.y, u.x);
    if (phi < 0) phi += 2.0 * M_PI;

    if (phi >= min_phi && phi <= max_phi) {
      const int bin_index = std::min(static_cast<int>(std::floor(((phi - min_phi) / (max_phi - min_phi)) * n_bins)), n_bins - 1);
      histogram[bin_index] += 1.0;
    }
  }

  return histogram;
}

AngularIntensity Detector::compute_speckle(const int n_theta, const int n_phi) const {
  AngularIntensity intensity(n_theta, n_phi, M_PI / 2.0, 2.0 * M_PI);
  CMatrix E_x(n_theta, n_phi), E_y(n_theta, n_phi), E_z(n_theta, n_phi);

  for (auto &ph : recorded_photons) {
    Vec3 u = ph.direction;

    // Get angles
    const double costtheta = -1 * u.z;
    const double theta = std::acos(costtheta);
    double phi = std::atan2(u.y, u.x);
    if (phi < 0) phi += 2.0 * M_PI;

    // Determine bins
    const int itheta = std::min(static_cast<int>(std::floor((theta / intensity.theta_max) * n_theta)), n_theta - 1);
    const int iphi   = std::min(static_cast<int>(std::floor((phi   / intensity.phi_max)   * n_phi)),   n_phi   - 1);

    // Compute local field contribution
    std::complex<double> phase = std::exp(std::complex<double>(0, ph.k * ph.opticalpath));
    std::complex<double> Em_local_photon = ph.polarization.m * phase * std::sqrt(ph.weight);
    std::complex<double> En_local_photon = ph.polarization.n * phase * std::sqrt(ph.weight);

    // Accumulate field contributions
    E_x(itheta, iphi) += Em_local_photon * m_polarization.x + En_local_photon * n_polarization.x;
    E_y(itheta, iphi) += Em_local_photon * m_polarization.y + En_local_photon * n_polarization.y;
    E_z(itheta, iphi) += Em_local_photon * m_polarization.z + En_local_photon * n_polarization.z;
  }

  // Precompute theta and phi edges for solid angle calculation
  std::vector<double> theta_edges(n_theta+1), phi_edges(n_phi+1);
  for (int it=0; it<=n_theta; ++it) theta_edges[it] = (intensity.theta_max * it) / n_theta;
  for (int ip=0; ip<=n_phi; ++ip) phi_edges[ip] = (intensity.phi_max * ip) / n_phi;

  // Compute intensities
  for (int it=0; it<n_theta; ++it) {
    const double th0 = theta_edges[it];
    const double th1 = theta_edges[it+1];
    const double dcos_theta = std::cos(th0) - std::cos(th1);

    for (int ip=0; ip<n_phi; ++ip) {
      const double ph0 = phi_edges[ip];
      const double ph1 = phi_edges[ip+1];
      const double dphi = ph1 - ph0;
      const double solid_angle = dcos_theta * dphi;
      // const double solid_angle = 1.0;

      intensity.Ix(it, ip) = std::norm(E_x(it, ip)) / solid_angle;
      intensity.Iy(it, ip) = std::norm(E_y(it, ip)) / solid_angle;
      intensity.Iz(it, ip) = std::norm(E_z(it, ip)) / solid_angle;
      intensity.I_total(it, ip) = intensity.Ix(it, ip) + intensity.Iy(it, ip) + intensity.Iz(it, ip);
      intensity.Ico(it, ip) = std::norm((E_x(it, ip) + std::complex<double>(0,1) * E_y(it, ip)) / std::sqrt(2.0)) / solid_angle;
      intensity.Icros(it, ip) = std::norm((E_x(it, ip) - std::complex<double>(0,1) * E_y(it, ip)) / std::sqrt(2.0)) / solid_angle;
    }
  }

  return intensity;
}

SpatialIntensity Detector::compute_spatial_intensity(const double x_len, const double y_len, const double max_theta, const int n_x, const int n_y) const {
  SpatialIntensity intensity(n_x, n_y, x_len, y_len);
  CMatrix E_x(n_x, n_y), E_y(n_x, n_y), E_z(n_x, n_y);

  const double min_x = -0.5 * x_len;
  const double min_y = -0.5 * y_len;
  const double max_x =  0.5 * x_len;
  const double max_y =  0.5 * y_len;

  const double area_per_bin = (x_len / n_x) * (y_len / n_y);
  uint photons = 0;

  for (auto &ph : recorded_photons) {
    Vec3 u = ph.direction;

    // Validate collection angle
    const double costtheta = -1 * u.z;
    const double theta = std::acos(costtheta);
    if (theta > max_theta) continue;

    // Validate position within detector area
    if (ph.position.x < min_x || ph.position.x >= max_x || ph.position.y < min_y || ph.position.y >= max_y) {
      continue;
    }

    // Determine bins
    const int ix = std::min(static_cast<int>(std::floor(((ph.position.x - min_x) / x_len) * n_x)), n_x - 1);
    const int iy = std::min(static_cast<int>(std::floor(((ph.position.y - min_y) / y_len) * n_y)), n_y - 1);

    // Compute local field contribution
    std::complex<double> phase = std::exp(std::complex<double>(0, ph.k * ph.opticalpath));
    std::complex<double> Em_local_photon = ph.polarization.m * phase * std::sqrt(ph.weight);
    std::complex<double> En_local_photon = ph.polarization.n * phase * std::sqrt(ph.weight);

    // Accumulate field contributions
    E_x(ix, iy) += Em_local_photon * m_polarization.x + En_local_photon * n_polarization.x;
    E_y(ix, iy) += Em_local_photon * m_polarization.y + En_local_photon * n_polarization.y;
    E_z(ix, iy) += Em_local_photon * m_polarization.z + En_local_photon * n_polarization.z;

    photons += 1;
  }

  LLOG_DEBUG("Computed spatial intensity from {} photons", photons);

  // Compute intensities
  for (int ix=0; ix<n_x; ++ix) {
    for (int iy=0; iy<n_y; ++iy) {
      intensity.Ix(ix, iy) = std::norm(E_x(ix, iy)) / area_per_bin;
      intensity.Iy(ix, iy) = std::norm(E_y(ix, iy)) / area_per_bin;
      intensity.Iz(ix, iy) = std::norm(E_z(ix, iy)) / area_per_bin;
      intensity.I_total(ix, iy) = intensity.Ix(ix, iy) + intensity.Iy(ix, iy) + intensity.Iz(ix, iy);
      intensity.Ico(ix, iy) = std::norm((E_x(ix, iy) + std::complex<double>(0,1) * E_y(ix, iy)) / std::sqrt(2.0)) / area_per_bin;
      intensity.Icros(ix, iy) = std::norm((E_x(ix, iy) - std::complex<double>(0,1) * E_y(ix, iy)) / std::sqrt(2.0)) / area_per_bin;
    }
  }

  return intensity;
}

AngularIntensity Detector::compute_angular_intensity(const double max_theta, const double max_phi, const int n_theta, const int n_phi) const {
  AngularIntensity intensity(n_theta, n_phi, max_theta, max_phi);
  CMatrix E_x(n_theta, n_phi), E_y(n_theta, n_phi), E_z(n_theta, n_phi);

  for (auto &ph : recorded_photons) {
    Vec3 u = ph.direction;

    // Validate collection angle
    const double costtheta = -1 * u.z;
    const double theta = std::acos(costtheta);
    if (theta > max_theta) continue;

    // Get angles
    double phi = std::atan2(u.y, u.x);
    if (phi < 0) phi += 2.0 * M_PI;

    // Determine bins
    const int itheta = std::min(static_cast<int>(std::floor((theta / intensity.theta_max) * n_theta)), n_theta - 1);
    const int iphi = std::min(static_cast<int>(std::floor((phi / intensity.phi_max) * n_phi)), n_phi - 1);

    // Compute local field contribution
    std::complex<double> phase = std::exp(std::complex<double>(0, ph.k * ph.opticalpath));
    std::complex<double> Em_local_photon = ph.polarization.m * phase * std::sqrt(ph.weight);
    std::complex<double> En_local_photon = ph.polarization.n * phase * std::sqrt(ph.weight);

    // Accumulate field contributions
    E_x(itheta, iphi) += Em_local_photon * m_polarization.x + En_local_photon * n_polarization.x;
    E_y(itheta, iphi) += Em_local_photon * m_polarization.y + En_local_photon * n_polarization.y;
    E_z(itheta, iphi) += Em_local_photon * m_polarization.z + En_local_photon * n_polarization.z;
  }

  // Precompute theta and phi edges for solid angle calculation
  std::vector<double> theta_edges(n_theta+1), phi_edges(n_phi+1);
  for (int it=0; it<=n_theta; ++it) theta_edges[it] = (intensity.theta_max * it) / n_theta;
  for (int ip=0; ip<=n_phi; ++ip) phi_edges[ip] = (intensity.phi_max * ip) / n_phi;

  // Compute intensities
  for (int it=0; it<n_theta; ++it) {
    const double th0 = theta_edges[it];
    const double th1 = theta_edges[it+1];
    const double dcos_theta = std::cos(th0) - std::cos(th1);

    for (int ip=0; ip<n_phi; ++ip) {
      const double ph0 = phi_edges[ip];
      const double ph1 = phi_edges[ip+1];
      const double dphi = ph1 - ph0;
      const double solid_angle = dcos_theta * dphi;

      intensity.Ix(it, ip) = std::norm(E_x(it, ip)) / solid_angle;
      intensity.Iy(it, ip) = std::norm(E_y(it, ip)) / solid_angle;
      intensity.Iz(it, ip) = std::norm(E_z(it, ip)) / solid_angle;
      intensity.I_total(it, ip) = intensity.Ix(it, ip) + intensity.Iy(it, ip) + intensity.Iz(it, ip);
      intensity.Ico(it, ip) = std::norm((E_x(it, ip) + std::complex<double>(0,1) * E_y(it, ip)) / std::sqrt(2.0)) / solid_angle;
      intensity.Icros(it, ip) = std::norm((E_x(it, ip) - std::complex<double>(0,1) * E_y(it, ip)) / std::sqrt(2.0)) / solid_angle;
    }
  }

  return intensity;
}

std::vector<SpatialIntensity> Detector::compute_time_resolved_spatial_intensity(const double x_len, const double y_len, const double max_theta, const double t_max, const double dt, const int n_x, const int n_y) const {
  LLOG_DEBUG("Starting computation of time-resolved spatial intensity:");

  const int n_time_bins = static_cast<int>(std::ceil(t_max / dt));
  std::vector<SpatialIntensity> time_resolved_intensities;
  time_resolved_intensities.reserve(n_time_bins);

  LLOG_DEBUG("Number of time bins: {}", n_time_bins);

  std::vector<CMatrix> E_x_time;
  std::vector<CMatrix> E_y_time;
  std::vector<CMatrix> E_z_time;
  E_x_time.reserve(n_time_bins);
  E_y_time.reserve(n_time_bins);
  E_z_time.reserve(n_time_bins);

  LLOG_DEBUG("Reserving space for field matrices for each time bin");

  for (int it=0; it<n_time_bins; ++it) {
    time_resolved_intensities.emplace_back(n_x, n_y, x_len, y_len);
    E_x_time.emplace_back(n_x, n_y);
    E_y_time.emplace_back(n_x, n_y);
    E_z_time.emplace_back(n_x, n_y);
  }

  LLOG_DEBUG("Computing time-resolved spatial intensity with {} time bins", n_time_bins);

  const double min_x = -0.5 * x_len;
  const double min_y = -0.5 * y_len;
  const double max_x =  0.5 * x_len;
  const double max_y =  0.5 * y_len;

  const double area_per_bin = (x_len / n_x) * (y_len / n_y);

  for (const auto &ph : recorded_photons) {
    Vec3 u = ph.direction;

    // Validate collection angle
    const double costtheta = -1 * u.z;
    const double theta = std::acos(costtheta);
    if (theta > max_theta) continue;

    // Validate position within detector area
    if (ph.position.x < min_x || ph.position.x >= max_x || ph.position.y < min_y || ph.position.y >= max_y) {
      continue;
    }

    // Determine time bin
    const int itime = static_cast<int>(ph.arrival_time / dt);
    if (itime < 0 || itime >= n_time_bins) {
      continue;
    }

    // Determine spatial bins
    const int ix = std::min(static_cast<int>(std::floor(((ph.position.x - min_x) / x_len) * n_x)), n_x - 1);
    const int iy = std::min(static_cast<int>(std::floor(((ph.position.y - min_y) / y_len) * n_y)), n_y - 1);

    // Compute local field contribution
    std::complex<double> phase = std::exp(std::complex<double>(0, ph.k * ph.opticalpath));
    std::complex<double> Em_local_photon = ph.polarization.m * phase * std::sqrt(ph.weight);
    std::complex<double> En_local_photon = ph.polarization.n * phase * std::sqrt(ph.weight);

    // Accumulate field contributions
    E_x_time[itime](ix, iy) += Em_local_photon * m_polarization.x + En_local_photon * n_polarization.x;
    E_y_time[itime](ix, iy) += Em_local_photon * m_polarization.y + En_local_photon * n_polarization.y;
    E_z_time[itime](ix, iy) += Em_local_photon * m_polarization.z + En_local_photon * n_polarization.z;
  }

  LLOG_DEBUG("Finished accumulating fields for all photons");

  // Compute intensities for each time bin
  for (int itime=0; itime<n_time_bins; ++itime) {
    auto &intensity = time_resolved_intensities[itime];
    const auto &E_x = E_x_time[itime];
    const auto &E_y = E_y_time[itime];
    const auto &E_z = E_z_time[itime];

    for (int ix=0; ix<n_x; ++ix) {
      for (int iy=0; iy<n_y; ++iy) {
        intensity.Ix(ix, iy) = std::norm(E_x(ix, iy)) / area_per_bin;
        intensity.Iy(ix, iy) = std::norm(E_y(ix, iy)) / area_per_bin;
        intensity.Iz(ix, iy) = std::norm(E_z(ix, iy)) / area_per_bin;
        intensity.I_total(ix, iy) = intensity.Ix(ix, iy) + intensity.Iy(ix, iy) + intensity.Iz(ix, iy);
        intensity.Ico(ix, iy) = std::norm((E_x(ix, iy) + std::complex<double>(0,1) * E_y(ix, iy)) / std::sqrt(2.0)) / area_per_bin;
        intensity.Icros(ix, iy) = std::norm((E_x(ix, iy) - std::complex<double>(0,1) * E_y(ix, iy)) / std::sqrt(2.0)) / area_per_bin;
      }
    }
  }

  return time_resolved_intensities;
}

void Detector::save_recorded_photons(const std::string &filename) const {
    // Abrir en modo binario y truncar (borrar contenido anterior)
    std::ofstream file(filename, std::ios::binary | std::ios::trunc);

    if (!file.is_open()) {
        // LLOG_ERROR("No se pudo abrir el archivo para guardar: {}", filename);
        throw std::runtime_error("No se pudo abrir el archivo para guardar: " + filename);
    }

    // 1. Escribir el número de fotones grabados.
    //    Esto es crucial para saber cuántos leer después.
    size_t num_photons = recorded_photons.size();
    file.write(reinterpret_cast<const char*>(&num_photons), sizeof(size_t));

    // 2. Escribir el bloque de datos de todos los fotones a la vez.
    if (num_photons > 0) {
        file.write(
            reinterpret_cast<const char*>(recorded_photons.data()),
            num_photons * sizeof(PhotonRecord)
        );
    }

    file.close();
}

void Detector::load_recorded_photons(const std::string &filename) {
    // Abrir en modo binario
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        // LLOG_ERROR("No se pudo abrir el archivo para cargar: {}", filename);
        throw std::runtime_error("No se pudo abrir el archivo para cargar: " + filename);
    }

    // 1. Leer el número de fotones que contiene el archivo.
    size_t num_photons = 0;
    file.read(reinterpret_cast<char*>(&num_photons), sizeof(size_t));

    if (!file) {
        // LLOG_ERROR("Error al leer el tamaño del archivo: {}", filename);
        throw std::runtime_error("Error al leer el tamaño del archivo: " + filename);
    }

    // Borra los fotones que pudieran estar en memoria
    recorded_photons.clear();

    if (num_photons > 0) {
        // 2. Preparar el vector para recibir los datos
        recorded_photons.resize(num_photons);

        // 3. Leer todo el bloque de datos directamente en la memoria del vector.
        file.read(
            reinterpret_cast<char*>(recorded_photons.data()),
            num_photons * sizeof(PhotonRecord)
        );
    }

    file.close();
}

} // namespace luminis::core
