#include <luminis/core/absortion.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/log/logger.hpp>
#include <cmath>

namespace luminis::core
{

  // ═══════════════════════════════════════════════════════════════════════════
  // Absorption — unified recorder (time-integrated + time-resolved)
  // ═══════════════════════════════════════════════════════════════════════════

  Absorption::Absorption(double r, double z, double dr, double dz, double dt, double t_max)
      : radius(r), depth(z), d_r(dr), d_z(dz), d_t(dt), t_max(t_max)
  {
    // Compute number of time bins
    if (d_t > 0.0)
      // Match timed sensor convention:
      //   bin 0 -> always time-integrated
      //   bins [1..n_t-1] -> explicit time windows of width d_t
      n_t = static_cast<int>(std::ceil(t_max / d_t)) + 1;
    else
      n_t = 1; // time-integrated

    // Compute spatial grid dimensions
    const std::size_t n_r = static_cast<std::size_t>(radius / d_r) + 1;
    const std::size_t n_z = static_cast<std::size_t>(depth / d_z) + 1;

    // Allocate zero-filled grids for each time bin
    time_slices.reserve(n_t);
    for (int i = 0; i < n_t; ++i)
      time_slices.emplace_back(n_r, n_z);
  }

  Absorption Absorption::clone() const
  {
    return Absorption(radius, depth, d_r, d_z, d_t, t_max);
  }

  void Absorption::merge_from(const Absorption &other)
  {
    for (int k = 0; k < n_t; ++k)
    {
      for (std::size_t i = 0; i < time_slices[k].rows; ++i)
      {
        for (std::size_t j = 0; j < time_slices[k].cols; ++j)
        {
          time_slices[k](i, j) += other.time_slices[k](i, j);
        }
      }
    }
  }

  void Absorption::record_absorption(const Photon &photon, double d_weight)
  {
    // Compute cylindrical (r, z) bin indices
    const Vec3 &pos = photon.pos;
    const double r = std::sqrt(pos.x * pos.x + pos.y * pos.y);
    const double z = pos.z;

    if (r > radius || z < 0.0 || z > depth)
      return;

    const std::size_t i_r = static_cast<std::size_t>(std::floor(r / d_r));
    const std::size_t i_z = static_cast<std::size_t>(std::floor(z / d_z));

    if (time_slices.empty())
      return;

    if (i_r >= time_slices[0].rows || i_z >= time_slices[0].cols)
    {
      LLOG_WARN("Calculated indices out of bounds: i_r={}, i_z={}", i_r, i_z);
      return;
    }

    // Bin 0 is always the integrated accumulation.
    if (!time_slices.empty())
      time_slices[0](i_r, i_z) += d_weight;

    // If time-resolved, also deposit into the corresponding time-window bin.
    // Bin mapping:
    //   k_window = floor(time / d_t)
    //   k_slice  = k_window + 1
    if (d_t > 0.0)
    {
      const double time = photon.launch_time + (photon.opticalpath / photon.velocity);
      int k = static_cast<int>(std::floor(time / d_t)) + 1;
      if (k >= 1 && k < n_t)
        time_slices[k](i_r, i_z) += d_weight;
    }
  }

  Matrix Absorption::get_absorption_image(int n_photons, int time_index) const
  {
    if (time_index < 0 || time_index >= n_t)
    {
      LLOG_ERROR("Time index out of bounds: {}", time_index);
      return {};
    }

    const Matrix &grid = time_slices[time_index];
    const int n_r = grid.rows;
    const int len_r = n_r * 2;
    const int len_z = grid.cols;
    Matrix image(len_r, len_z);

    for (int i = 0; i < n_r; ++i)
    {
      const double r_in  = i * d_r;
      const double r_out = (i + 1) * d_r;
      const double ring_area = M_PI * (r_out * r_out - r_in * r_in);
      const double voxel_vol = ring_area * d_z;

      for (int j = 0; j < len_z; ++j)
      {
        const double per_vol = grid(i, j) / (static_cast<double>(n_photons) * voxel_vol);
        image(n_r - i, j) = per_vol;
        if (i != 0)
          image(n_r + i, j) = per_vol;
      }
    }
    return image;
  }

  Matrix Absorption::get_total_image(int n_photons) const
  {
    return get_absorption_image(n_photons, 0);
  }

} // namespace luminis::core
