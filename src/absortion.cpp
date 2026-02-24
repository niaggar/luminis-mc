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
      n_t = static_cast<int>(std::ceil(t_max / d_t));
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
    // Determine which time bin to deposit into
    int k = 0;
    if (d_t > 0.0)
    {
      const double time = photon.launch_time + (photon.opticalpath / photon.velocity);
      k = static_cast<int>(std::floor(time / d_t));
      if (k < 0 || k >= n_t)
        return;
    }

    // Compute cylindrical (r, z) bin indices
    const Vec3 &pos = photon.pos;
    const double r = std::sqrt(pos.x * pos.x + pos.y * pos.y);
    const double z = pos.z;

    if (r > radius || z < 0.0 || z > depth)
      return;

    const std::size_t i_r = static_cast<std::size_t>(std::floor(r / d_r));
    const std::size_t i_z = static_cast<std::size_t>(std::floor(z / d_z));

    Matrix &grid = time_slices[k];
    if (i_r >= grid.rows || i_z >= grid.cols)
    {
      LLOG_WARN("Calculated indices out of bounds: i_r={}, i_z={}", i_r, i_z);
      return;
    }

    grid(i_r, i_z) += d_weight;
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

  std::optional<Absorption> combine_absorptions(const std::vector<Absorption> &absorptions)
  {
    if (absorptions.empty())
    {
      LLOG_ERROR("No absorption data to combine.");
      return std::nullopt;
    }

    const auto &ref = absorptions[0];
    Absorption combined(ref.radius, ref.depth, ref.d_r, ref.d_z, ref.d_t, ref.t_max);

    for (const auto &abs : absorptions)
    {
      if (abs.radius != ref.radius || abs.depth != ref.depth ||
          abs.d_r != ref.d_r || abs.d_z != ref.d_z ||
          abs.d_t != ref.d_t || abs.n_t != ref.n_t)
      {
        LLOG_ERROR("Inconsistent absorption configurations found during combination.");
        return std::nullopt;
      }

      for (int k = 0; k < combined.n_t; ++k)
      {
        for (std::size_t i = 0; i < combined.time_slices[k].rows; ++i)
        {
          for (std::size_t j = 0; j < combined.time_slices[k].cols; ++j)
          {
            combined.time_slices[k](i, j) += abs.time_slices[k](i, j);
          }
        }
      }
    }

    return combined;
  }

} // namespace luminis::core
