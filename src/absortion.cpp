#include <luminis/core/detector.hpp>
#include <luminis/core/absortion.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/log/logger.hpp>

namespace luminis::core
{

  Absorption::Absorption(double r, double z, double dr, double dz)
      : radius(r), depth(z), d_r(dr), d_z(dz)
  {
    const std::size_t n_r = static_cast<std::size_t>(radius / d_r) + 1;
    const std::size_t n_z = static_cast<std::size_t>(depth / d_z) + 1;
    absorption_values = Matrix(n_r, n_z);
  }

  void Absorption::record_absorption(const Photon &photon, double d_weight)
  {
    const Vec3 &pos = photon.pos;
    const double r = std::sqrt(pos.x * pos.x + pos.y * pos.y);
    const double z = pos.z;

    if (r > radius || z < 0 || z > depth)
    {
      return;
    }

    const std::size_t i_r = static_cast<std::size_t>(std::floor(r / d_r));
    const std::size_t i_z = static_cast<std::size_t>(std::floor(z / d_z));

    if (i_r >= absorption_values.rows || i_z >= absorption_values.cols)
    {
      LLOG_WARN("Calculated indices out of bounds: i_r={}, i_z={}", i_r, i_z);
      return;
    }

    absorption_values(i_r, i_z) += d_weight;
  }
  
  Matrix Absorption::get_absorption_image(const int n_photons) const
  {
    const int n_r = absorption_values.rows;
    const int len_r = (n_r * 2);
    const int len_z = absorption_values.cols;
    Matrix image(len_r, len_z);

    for (int i = 0; i < n_r; ++i) {
      const double r_in  = i * d_r;
      const double r_out = (i + 1) * d_r;
      const double ring_area = M_PI * (r_out*r_out - r_in*r_in);     // área anular
      const double voxel_vol = ring_area * d_z;                      // volumen del voxel cilíndrico

      for (int j = 0; j < len_z; ++j) {
        const double per_vol = absorption_values(i, j) / (static_cast<double>(n_photons) * voxel_vol);
        // const double log_value = std::log10(per_vol + 1e-20);        // evita -inf
        image(n_r - i, j) = per_vol;          // mitad superior
        if (i != 0) image(n_r + i, j) = per_vol; // mitad inferior
      }
    }
    return image;
  }

  AbsorptionTimeDependent::AbsorptionTimeDependent(double r, double z, double dr, double dz, double dt, double t_max)
  {
    radius = r;
    depth = z;
    d_r = dr;
    d_z = dz;
    d_t = dt;

    if (d_t > 0)
    {
      n_t_slices = static_cast<int>(std::ceil(t_max / d_t));
    }
    else
    {
      n_t_slices = 1; // Single slice for time-independent case
    }
    this->t_max = t_max;

    if (d_t <= 0.0)
    { // sin tiempo: un solo slice
      time_slices.reserve(1);
      time_slices.push_back(Absorption(radius, depth, d_r, d_z));
    }
    else
    {
      time_slices.reserve(n_t_slices + 1);
      for (int i = 0; i < n_t_slices; ++i)
      {
        time_slices.push_back(Absorption(radius, depth, d_r, d_z));
      }
    }
  }

  AbsorptionTimeDependent AbsorptionTimeDependent::clone() const
  {
    return AbsorptionTimeDependent(radius, depth, d_r, d_z, d_t, t_max);
  }

  void AbsorptionTimeDependent::merge_from(const AbsorptionTimeDependent &other)
  {
    for (int k = 0; k < n_t_slices; ++k)
    {
      for (std::size_t i = 0; i < time_slices[k].absorption_values.rows; ++i)
      {
        for (std::size_t j = 0; j < time_slices[k].absorption_values.cols; ++j)
        {
          time_slices[k].absorption_values(i, j) += other.time_slices[k].absorption_values(i, j);
        }
      }
    }
  }

  void AbsorptionTimeDependent::record_absorption(const Photon &photon, double d_weight)
  {
    if (d_t == 0)
    {
      time_slices[0].record_absorption(photon, d_weight);
      return;
    }

    const double time = photon.launch_time + (photon.opticalpath / photon.velocity); // in ns
    const std::size_t k = static_cast<std::size_t>(std::floor(time / d_t));
    if (k >= time_slices.size())
    {
      return;
    }
    time_slices[k].record_absorption(photon, d_weight);
  }
  
  Matrix AbsorptionTimeDependent::get_absorption_image(const int n_photons, const int time_index) const
  {
    if (time_index < 0 || time_index >= time_slices.size())
    {
      LLOG_ERROR("Time index out of bounds: {}", time_index);
      return {};
    }
    return time_slices[time_index].get_absorption_image(n_photons);
  }

  AbsorptionTimeDependent *combine_absorptions(const std::vector<AbsorptionTimeDependent> &absorptions)
  {
    if (absorptions.empty())
    {
      LLOG_ERROR("No absorption data to combine.");
      return nullptr;
    }

    const double radius = absorptions[0].radius;
    const double depth = absorptions[0].depth;
    const double d_r = absorptions[0].d_r;
    const double d_z = absorptions[0].d_z;
    const double d_t = absorptions[0].d_t;
    const int n_t_slices = absorptions[0].n_t_slices;

    AbsorptionTimeDependent *combined = new AbsorptionTimeDependent(radius, depth, d_r, d_z, d_t, n_t_slices * d_t);

    for (const auto &abs : absorptions)
    {
      if (abs.radius != radius || abs.depth != depth || abs.d_r != d_r || abs.d_z != d_z || abs.d_t != d_t || abs.n_t_slices != n_t_slices)
      {
        LLOG_ERROR("Inconsistent absorption configurations found during combination.");
        delete combined;
        return nullptr;
      }

      for (int k = 0; k < n_t_slices; ++k)
      {
        for (std::size_t i = 0; i < combined->time_slices[k].absorption_values.rows; ++i)
        {
          for (std::size_t j = 0; j < combined->time_slices[k].absorption_values.cols; ++j)
          {
            combined->time_slices[k].absorption_values(i, j) += abs.time_slices[k].absorption_values(i, j);
          }
        }
      }
    }

    return combined;
  }

} // namespace luminis::core
