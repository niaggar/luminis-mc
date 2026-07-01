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
  //  Layer / HomogeneousLayer
  // ══════════════════════════════════════════════════════════════════════════════

  bool Layer::contains(double z) const
  {
    return z >= z_min && z < z_max;
  }

  double Layer::thickness() const
  {
    return z_max - z_min;
  }

  HomogeneousLayer::HomogeneousLayer(const ScatteringMedium *medium, double z_min, double z_max)
      : Layer(z_min, z_max), medium(medium) {}

  MixtureLayer::MixtureLayer(const std::vector<const ScatteringMedium *> &species,
                             const std::vector<double> &number_densities,
                             double z_min, double z_max)
      : Layer(z_min, z_max), species(species), number_densities(number_densities)
  {
    if (species.empty())
      throw std::invalid_argument("MixtureLayer: species list must not be empty.");
    if (species.size() != number_densities.size())
      throw std::invalid_argument("MixtureLayer: species and number_densities sizes differ.");

    mu_s_i.resize(species.size());
    mu_s_total = 0.0;
    mu_a_total = 0.0;
    for (std::size_t i = 0; i < species.size(); ++i)
    {
      const ScatteringMedium *sp = species[i];
      if (!sp)
        throw std::invalid_argument("MixtureLayer: species pointer must not be null.");

      // μ_s^(i) = n_i · σ_s^(i): density × single-particle scattering cross-section.
      const double mu_s = number_densities[i] * sp->scattering_cross_section();
      mu_s_i[i] = mu_s;
      mu_s_total += mu_s;

      // Aggregate absorption preserves each species' single-scattering albedo:
      // μ_a^(i) = μ_s^(i) · (μ_a/μ_s)_species  (0 if that species has μ_s = 0).
      if (sp->mu_scattering > 0.0)
        mu_a_total += mu_s * (sp->mu_absorption / sp->mu_scattering);
    }

    if (mu_s_total <= 0.0)
      throw std::invalid_argument(
          "MixtureLayer: total μ_s must be positive (check number densities and "
          "that the species' phase functions expose a scattering cross-section).");

    mu_t_total = mu_s_total + mu_a_total;
    mfp_total = 1.0 / mu_s_total;

    // Normalized selection CDF over μ_s^(i) (ends exactly at 1 on the last entry).
    selection_cdf.resize(species.size());
    double acc = 0.0;
    for (std::size_t i = 0; i < species.size(); ++i)
    {
      acc += mu_s_i[i] / mu_s_total;
      selection_cdf[i] = acc;
    }
    selection_cdf.back() = 1.0; // guard against round-off so lower_bound always hits.
  }

  double MixtureLayer::sample_free_path(luminis::math::Rng &rng) const
  {
    // Exponential with the aggregate mean free path (same form as the media).
    return -1 * mfp_total * std::log(rng.uniform());
  }

  const ScatteringMedium *MixtureLayer::select_scatter_medium(luminis::math::Rng &rng) const
  {
    const double u = rng.uniform();
    auto it = std::lower_bound(selection_cdf.begin(), selection_cdf.end(), u);
    std::size_t idx = static_cast<std::size_t>(it - selection_cdf.begin());
    if (idx >= species.size())
      idx = species.size() - 1; // numerical guard for u == 1.0
    return species[idx];
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

  // Validate that a new [z_min, z_max) slab is contiguous with the current top
  // of the stack (shared by add_layer and add_mixture_layer).
  static void validate_contiguous(const std::vector<std::unique_ptr<Layer>> &layers,
                                  double z_min, double z_max)
  {
    if (z_max <= z_min)
      throw std::invalid_argument("Sample::add_layer: z_max must be greater than z_min.");

    if (layers.empty())
    {
      if (z_min != 0.0)
        throw std::invalid_argument("Sample::add_layer: first layer must start at z = 0.");
      return;
    }

    const double prev_z_max = layers.back()->z_max;
    if (std::isinf(prev_z_max))
      throw std::invalid_argument("Sample::add_layer: cannot add a layer after an infinite top layer.");
    if (z_min != prev_z_max)
      throw std::invalid_argument(
          "Sample::add_layer: layer z_min (" + std::to_string(z_min) +
          ") does not match previous z_max (" + std::to_string(prev_z_max) + ").");
  }

  void Sample::add_layer(const ScatteringMedium *medium, double z_min, double z_max)
  {
    if (!medium)
    {
      throw std::invalid_argument("Sample::add_layer: medium pointer must not be null.");
    }
    validate_contiguous(layers, z_min, z_max);

    layers.push_back(std::make_unique<HomogeneousLayer>(medium, z_min, z_max));

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

  void Sample::add_mixture_layer(const std::vector<const ScatteringMedium *> &species,
                                 const std::vector<double> &number_densities,
                                 double z_min, double z_max)
  {
    validate_contiguous(layers, z_min, z_max);

    // MixtureLayer's constructor validates the species/density inputs and
    // computes the aggregate coefficients and selection CDF.
    auto layer = std::make_unique<MixtureLayer>(species, number_densities, z_min, z_max);
    const double mu_s = layer->mu_s_total;
    const double mu_a = layer->mu_a_total;
    layers.push_back(std::move(layer));

    if (layers.size() > 1)
    {
      interfaces.push_back(z_min);
    }

    LLOG_DEBUG("Sample: added mixture layer {} with z ∈ [{}, {}), {} species, μ_s={}, μ_a={}",
               layers.size() - 1, z_min, z_max, species.size(), mu_s, mu_a);
  }

  std::size_t Sample::size() const
  {
    return layers.size();
  }

  const Layer &Sample::get_layer(std::size_t index) const
  {
    return *layers.at(index);
  }

  const Layer *Sample::get_layer_at(double z) const
  {
    if (layers.empty())
      return nullptr;
    if (z < 0.0)
      return nullptr;

    // Check top boundary.
    if (!std::isinf(layers.back()->z_max) && z >= layers.back()->z_max)
      return nullptr;

    // Binary search: find the first interface > z.
    // `interfaces` stores the z_min of layers[1..N-1].
    // The layer index is: (number of interfaces <= z).
    auto it = std::upper_bound(interfaces.begin(), interfaces.end(), z);
    std::size_t idx = static_cast<std::size_t>(it - interfaces.begin());

    // idx is the index into layers (0-based): layer 0 covers [0, interfaces[0]),
    // layer 1 covers [interfaces[0], interfaces[1]), etc.
    if (idx < layers.size())
      return layers[idx].get();
    return nullptr;
  }

  std::size_t Sample::get_layer_index_at(double z) const
  {
    if (layers.empty())
      return layers.size();
    if (z < 0.0)
      return layers.size();
    if (!std::isinf(layers.back()->z_max) && z >= layers.back()->z_max)
      return layers.size();

    auto it = std::upper_bound(interfaces.begin(), interfaces.end(), z);
    std::size_t idx = static_cast<std::size_t>(it - interfaces.begin());
    if (idx < layers.size())
      return idx;
    return layers.size();
  }

  std::optional<double> Sample::find_next_interface(double z_current, double z_target) const
  {
    if (interfaces.empty())
      return std::nullopt;

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
    if (layers.empty())
      return false;

    // NaN or Inf check.
    if (std::isnan(position.x) || std::isnan(position.y) || std::isnan(position.z))
      return false;
    if (std::isinf(position.x) || std::isinf(position.y) || std::isinf(position.z))
      return false;

    // Bottom boundary: z >= 0.
    if (position.z < 0.0)
      return false;

    // Top boundary: z < z_max of last layer (unless infinite).
    if (!std::isinf(layers.back()->z_max) && position.z >= layers.back()->z_max)
      return false;

    return true;
  }

  double Sample::z_top() const
  {
    if (layers.empty())
      return 0.0;
    return layers.back()->z_max;
  }

} // namespace luminis::core
