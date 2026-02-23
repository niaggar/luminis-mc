/**
 * @file sample.cpp
 * @brief Implementation of SampleLayer and Sample.
 *
 * Provides the layer container that manages multiple scattering media
 * arranged as contiguous z-slabs ("cake layers"). The binary search
 * on the sorted `interfaces` vector gives O(log N) layer lookups.
 *
 * The Sample also manages the shared host medium refractive index and
 * derived photon velocity, which are uniform across all layers.
 *
 * @see include/luminis/core/sample.hpp
 */

#include <luminis/core/sample.hpp>
#include <luminis/log/logger.hpp>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace luminis::core
{

// ══════════════════════════════════════════════════════════════════════════════
//  SampleLayer
// ══════════════════════════════════════════════════════════════════════════════

  SampleLayer::SampleLayer(const ScatteringMedium *medium, double z_min, double z_max)
      : medium(medium), z_min(z_min), z_max(z_max) {}

  bool SampleLayer::contains(double z) const
  {
    return z >= z_min && z < z_max;
  }

  double SampleLayer::thickness() const
  {
    return z_max - z_min;
  }

// ══════════════════════════════════════════════════════════════════════════════
//  Sample
// ══════════════════════════════════════════════════════════════════════════════

  Sample::Sample(double n_medium)
      : refractive_index(n_medium), light_speed(1.0 / n_medium) {}

  double Sample::light_speed_in_medium() const
  {
    return light_speed;
  }

  void Sample::add_layer(const ScatteringMedium *medium, double z_min, double z_max)
  {
    if (!medium)
    {
      throw std::invalid_argument("Sample::add_layer: medium pointer must not be null.");
    }
    if (z_max <= z_min)
    {
      throw std::invalid_argument("Sample::add_layer: z_max must be greater than z_min.");
    }

    if (layers.empty())
    {
      // First layer must start at z = 0.
      if (z_min != 0.0)
      {
        throw std::invalid_argument("Sample::add_layer: first layer must start at z = 0.");
      }
    }
    else
    {
      // Subsequent layers must be contiguous.
      const double prev_z_max = layers.back().z_max;
      if (std::isinf(prev_z_max))
      {
        throw std::invalid_argument("Sample::add_layer: cannot add a layer after an infinite top layer.");
      }
      if (z_min != prev_z_max)
      {
        throw std::invalid_argument(
            "Sample::add_layer: layer z_min (" + std::to_string(z_min) +
            ") does not match previous z_max (" + std::to_string(prev_z_max) + ").");
      }
    }

    layers.emplace_back(medium, z_min, z_max);

    // Record interface z-coordinates (boundaries between layers).
    // The first layer's z_min = 0 is the bottom boundary (handled by is_inside).
    // Each subsequent layer's z_min is an internal interface.
    if (layers.size() > 1)
    {
      interfaces.push_back(z_min);
    }

    LLOG_DEBUG("Sample: added layer {} with z ∈ [{}, {}), μ_s={}, μ_a={}",
               layers.size() - 1, z_min, z_max,
               medium->mu_scattering, medium->mu_absorption);
  }

  std::size_t Sample::size() const
  {
    return layers.size();
  }

  const SampleLayer& Sample::get_layer(std::size_t index) const
  {
    return layers.at(index);
  }

  const SampleLayer* Sample::get_layer_at(double z) const
  {
    if (layers.empty()) return nullptr;
    if (z < 0.0) return nullptr;

    // Check top boundary.
    if (!std::isinf(layers.back().z_max) && z >= layers.back().z_max)
      return nullptr;

    // Binary search: find the first interface > z.
    // `interfaces` stores the z_min of layers[1..N-1].
    // The layer index is: (number of interfaces <= z).
    auto it = std::upper_bound(interfaces.begin(), interfaces.end(), z);
    std::size_t idx = static_cast<std::size_t>(it - interfaces.begin());

    // idx is the index into layers (0-based): layer 0 covers [0, interfaces[0]),
    // layer 1 covers [interfaces[0], interfaces[1]), etc.
    if (idx < layers.size())
      return &layers[idx];
    return nullptr;
  }

  std::size_t Sample::get_layer_index_at(double z) const
  {
    const SampleLayer* layer = get_layer_at(z);
    if (!layer) return layers.size();

    // Find the index by pointer arithmetic.
    return static_cast<std::size_t>(layer - layers.data());
  }

  std::optional<double> Sample::find_next_interface(double z_current, double z_target) const
  {
    if (interfaces.empty()) return std::nullopt;

    if (z_target > z_current)
    {
      // Moving upward (increasing z): find the smallest interface > z_current
      // that is <= z_target.
      auto it = std::upper_bound(interfaces.begin(), interfaces.end(), z_current);
      if (it != interfaces.end() && *it <= z_target)
        return *it;
    }
    else if (z_target < z_current)
    {
      // Moving downward (decreasing z): find the largest interface <= z_current
      // that is > z_target.
      // lower_bound gives first element >= z_current.
      auto it = std::lower_bound(interfaces.begin(), interfaces.end(), z_current);
      if (it != interfaces.begin())
      {
        --it;
        // *it < z_current. Check if *it >= z_target.
        if (*it >= z_target)
          return *it;
      }
    }

    return std::nullopt;
  }

  bool Sample::is_inside(const Vec3 &position) const
  {
    if (layers.empty()) return false;

    // NaN or Inf check.
    if (std::isnan(position.x) || std::isnan(position.y) || std::isnan(position.z))
      return false;
    if (std::isinf(position.x) || std::isinf(position.y) || std::isinf(position.z))
      return false;

    // Bottom boundary: z >= 0.
    if (position.z < 0.0)
      return false;

    // Top boundary: z < z_max of last layer (unless infinite).
    if (!std::isinf(layers.back().z_max) && position.z >= layers.back().z_max)
      return false;

    return true;
  }

  double Sample::z_top() const
  {
    if (layers.empty()) return 0.0;
    return layers.back().z_max;
  }

} // namespace luminis::core
