#include <luminis/core/detector.hpp>
#include <luminis/core/absortion.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/log/logger.hpp>

namespace luminis::core {

Absorption::Absorption(double r, double z, double dr, double dz) {
  radius = r;
  depth = z;
  d_r = dr;
  d_z = dz;

  const std::size_t n_r = static_cast<std::size_t>(radius / d_r) + 1;
  const std::size_t n_z = static_cast<std::size_t>(depth / d_z) + 1;

  absorption_values.resize(n_r, std::vector<double>(n_z, 0.0));
}

void Absorption::record_absorption(const Photon &photon, double d_weight) {
  const Vec3 &pos = photon.pos;
  const double r = std::sqrt(pos.x * pos.x + pos.y * pos.y);
  const double z = pos.z;

  if (r < 0 || r > radius || z < 0 || z > depth) {
    return;
  }

  const std::size_t i_r = static_cast<std::size_t>(std::floor(r / d_r));
  const std::size_t i_z = static_cast<std::size_t>(std::floor(z / d_z));

  if (i_r >= absorption_values.size() || i_z >= absorption_values[0].size()) {
    LLOG_WARN("Calculated indices out of bounds: i_r={}, i_z={}", i_r, i_z);
    return;
  }

  // absorption_values[i_r][i_z] += 1.0;
  absorption_values[i_r][i_z] += d_weight;
}
std::vector<std::vector<double>> Absorption::get_absorption_image(const int n_photons) const {
  const int n_r = absorption_values.size();
  const int len_r = (n_r * 2);
  const int len_z = absorption_values[0].size();
  std::vector<std::vector<double>> image(len_r, std::vector<double>(len_z, 0.0));

  for (int i = 0; i < n_r; ++i) {
    const double r_in  = i * d_r;
    const double r_out = (i + 1) * d_r;
    const double ring_area = M_PI * (r_out*r_out - r_in*r_in);     // área anular
    const double voxel_vol = ring_area * d_z;                      // volumen del voxel cilíndrico

    for (int j = 0; j < len_z; ++j) {
      const double per_vol = absorption_values[i][j] / (static_cast<double>(n_photons) * voxel_vol);
      const double log_value = std::log10(per_vol + 1e-20);        // evita -inf
      image[n_r - i][j] = log_value;          // mitad superior
      if (i != 0) image[n_r + i][j] = log_value; // mitad inferior
    }
  }
  return image;
}

AbsorptionTimeDependent::AbsorptionTimeDependent(double r, double z, double dr, double dz, double dt, double t_max) {
  radius = r;
  depth = z;
  d_r = dr;
  d_z = dz;
  d_t = dt;

  if (d_t <= 0.0) { // sin tiempo: un solo slice
    time_slices.emplace_back(r, z, dr, dz);
    return;
  }
  if (t_max > 0.0) {
    const std::size_t n = static_cast<std::size_t>(std::floor(t_max/d_t)) + 1;
    time_slices.assign(n, Absorption(r, z, dr, dz));
  } else {
    time_slices.clear();
  }
}
void AbsorptionTimeDependent::record_absorption(const Photon &photon, double d_weight) {
  if (d_t == 0) {
    time_slices[0].record_absorption(photon, d_weight);
    return;
  }

  const double time = photon.launch_time + (photon.opticalpath / photon.velocity); // in ns
  const std::size_t k = static_cast<std::size_t>(std::floor(time / d_t));
  if (k >= time_slices.size()) {
    time_slices.resize(k + 1, Absorption(radius, depth, d_r, d_z));
  }
  time_slices[k].record_absorption(photon, d_weight);
}
std::vector<std::vector<double>> AbsorptionTimeDependent::get_absorption_image(const int n_photons, const int time_index) const {
  if (time_index < 0 || time_index >= time_slices.size()) {
    LLOG_ERROR("Time index out of bounds: {}", time_index);
    return {};
  }
  return time_slices[time_index].get_absorption_image(n_photons);
}

} // namespace luminis::core
