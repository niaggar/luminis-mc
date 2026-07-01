/**
 * @file sample.hpp
 * @brief Layered sample container for Monte Carlo photon transport.
 *
 * Defines two types:
 *
 * - **SampleLayer**: associates a `ScatteringMedium*` with a z-range
 *   `[z_min, z_max)`. Each layer owns a non-owning pointer to a
 *   ScatteringMedium and defines the vertical slab in which that medium's
 *   optical properties apply.
 *
 * - **Sample**: an ordered collection of contiguous `SampleLayer` entries
 *   forming a "cake" of scattering media. Layers are sorted by z_min and must
 *   be contiguous (no gaps, no overlaps). The bottom boundary is always z = 0;
 *   the top layer may extend to z = +∞.
 *
 * The Sample also stores the **shared host medium refractive index** `n_medium`
 * and derived `light_speed`. All layers share the same background solvent;
 * only the particle properties (scattering/absorption coefficients, phase
 * function, etc.) differ between layers. Because the refractive index is
 * uniform across all layers, there is no Fresnel reflection/refraction at
 * layer interfaces — photons simply transition with updated scattering
 * properties.
 *
 * During transport, the simulation queries the sample for:
 *   - The layer containing a given z-coordinate (`get_layer_at`)
 *   - The next interface crossed by a photon step (`find_next_interface`)
 *   - Whether a position is inside the sample volume (`is_inside`)
 *   - The photon velocity via `light_speed_in_medium()`
 *
 * @see medium.hpp     — ScatteringMedium base class and concrete media
 * @see simulation.hpp — transport loop consuming the Sample
 */

#pragma once
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <vector>
#include <luminis/core/medium.hpp>

namespace luminis::core {

// ══════════════════════════════════════════════════════════════════════════════
//  Layer — abstract slab of the layered sample
// ══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Abstract slab of a layered sample.
 *
 * A `Layer` owns a half-open z-range `[z_min, z_max)` and exposes the
 * transport interface the per-event kernel consumes. Concrete layers decide
 * how a free path is sampled, what the aggregate optical coefficients are, and
 * which `ScatteringMedium` an individual scattering event is resolved against.
 *
 * The z-geometry (`contains`, `thickness`) and the Sample-level navigation
 * (`find_next_interface`, `is_inside`, `get_layer_index_at`,
 * `light_speed_in_medium`) depend only on `z_min`/`z_max` and the shared host
 * index — they never touch the scattering medium.
 *
 * Concrete types:
 *   - `HomogeneousLayer`: a single co-located `ScatteringMedium` (the historical
 *     one-species-per-depth behaviour).
 *   - `MixtureLayer` (added later): several co-located species, one chosen per
 *     event with probability μ_s^(i) / Σ μ_s.
 */
struct Layer {
  double z_min{0.0};                       ///< Lower z-boundary of the layer [mm] (inclusive).
  double z_max{0.0};                       ///< Upper z-boundary of the layer [mm] (exclusive, or +∞).

  Layer() = default;
  Layer(double z_min, double z_max) : z_min(z_min), z_max(z_max) {}
  virtual ~Layer() = default;

  /**
   * @brief Test whether a z-coordinate lies within this layer.
   * @param z  z-coordinate [mm].
   * @return   `true` if `z_min <= z < z_max`.
   */
  bool contains(double z) const;

  /**
   * @brief Return the thickness of the layer.
   *
   * Returns `+∞` if `z_max` is infinite.
   * @return Thickness [mm].
   */
  double thickness() const;

  // ── Transport interface consumed by the per-event kernel ───────────────────

  /// @brief Sample a free-path length for a step starting in this layer.
  virtual double sample_free_path(luminis::math::Rng &rng) const = 0;

  /// @brief Aggregate extinction coefficient μ_t = μ_a + μ_s of the layer [1/mm].
  virtual double mu_attenuation() const = 0;

  /// @brief Aggregate scattering coefficient μ_s of the layer [1/mm].
  virtual double mu_scattering() const = 0;

  /// @brief Aggregate absorption coefficient μ_a of the layer [1/mm].
  virtual double mu_absorption() const = 0;

  /**
   * @brief Resolve which scattering species this event scatters against.
   *
   * For a single-species (`HomogeneousLayer`) this returns the lone medium and
   * draws nothing from `rng` (preserving bit-for-bit reproducibility of the
   * single-species path). For a mixture it draws one species with probability
   * μ_s^(i)/Σ μ_s.
   */
  virtual const ScatteringMedium *select_scatter_medium(luminis::math::Rng &rng) const = 0;
};

// ══════════════════════════════════════════════════════════════════════════════
//  HomogeneousLayer — one slab backed by a single scattering medium
// ══════════════════════════════════════════════════════════════════════════════

/**
 * @brief A layer backed by a single `ScatteringMedium`.
 *
 * Associates a non-owning pointer to a `ScatteringMedium` with a half-open
 * z-range `[z_min, z_max)`. For the topmost layer, `z_max` may be `+∞`
 * (`std::numeric_limits<double>::infinity()`), indicating an unbounded
 * semi-infinite slab. All transport accessors delegate to the wrapped medium;
 * `select_scatter_medium` returns it without consuming any random draw.
 */
struct HomogeneousLayer : public Layer {
  const ScatteringMedium *medium{nullptr}; ///< Non-owning pointer to the scattering medium for this layer.

  /**
   * @brief Construct a HomogeneousLayer.
   *
   * @param medium  Pointer to the scattering medium (must remain valid for the sample lifetime).
   * @param z_min   Lower z-boundary [mm].
   * @param z_max   Upper z-boundary [mm] (use INFINITY for unbounded top).
   */
  HomogeneousLayer(const ScatteringMedium *medium, double z_min, double z_max);

  double sample_free_path(luminis::math::Rng &rng) const override { return medium->sample_free_path(rng); }
  double mu_attenuation() const override { return medium->mu_attenuation; }
  double mu_scattering() const override { return medium->mu_scattering; }
  double mu_absorption() const override { return medium->mu_absorption; }

  /// @brief Always the single wrapped medium; draws nothing from `rng`.
  const ScatteringMedium *select_scatter_medium(luminis::math::Rng & /*rng*/) const override { return medium; }
};

// ══════════════════════════════════════════════════════════════════════════════
//  MixtureLayer — several co-located species in one slab
// ══════════════════════════════════════════════════════════════════════════════

/**
 * @brief A layer holding several co-located scattering species (a mixture).
 *
 * All species occupy the same z-range `[z_min, z_max)`. Each scattering event
 * is resolved against one species j drawn with probability
 * `p_j = μ_s^(j) / Σ_i μ_s^(i)`, with `μ_s^(i) = n_i · σ_s^(i)` (number density
 * × single-particle scattering cross-section). The dilute / independent-scatter
 * regime is assumed: no structure factor, the species simply add their μ_s.
 *
 * Aggregate transport coefficients are precomputed in the constructor:
 *   - `mu_s_total = Σ_i n_i σ_s^(i)`
 *   - `mu_a_total = Σ_i μ_s^(i) · (a_i)` with each species' own absorption-to-
 *     scattering ratio `a_i = μ_a^(i)/μ_s^(i)` (preserves each species' single-
 *     scattering albedo; zero when a species has μ_s^(i)=0 or μ_a^(i)=0).
 *   - `mu_t_total = mu_s_total + mu_a_total`,  `mfp_total = 1/mu_s_total`.
 * The selection CDF over `μ_s^(i)` is normalized to end at 1.
 */
struct MixtureLayer : public Layer {
  std::vector<const ScatteringMedium *> species;  ///< Co-located species (non-owning).
  std::vector<double> number_densities;           ///< n_i [1/mm³] per species.
  std::vector<double> mu_s_i;                      ///< μ_s^(i) = n_i · σ_s^(i) [1/mm].
  std::vector<double> selection_cdf;               ///< Normalized CDF of μ_s^(i) (ends at 1).

  double mu_s_total{0.0}; ///< Σ μ_s^(i) [1/mm].
  double mu_a_total{0.0}; ///< Aggregate absorption [1/mm].
  double mu_t_total{0.0}; ///< μ_s_total + μ_a_total [1/mm].
  double mfp_total{0.0};  ///< 1 / μ_s_total [mm].

  /**
   * @brief Construct a mixture from species and their number densities.
   *
   * @param species           Co-located scattering media (must outlive the sample).
   * @param number_densities  Number density n_i [1/mm³] for each species.
   * @param z_min             Lower z-boundary [mm].
   * @param z_max             Upper z-boundary [mm] (use INFINITY for unbounded top).
   *
   * @throws std::invalid_argument on size mismatch, empty input, or μ_s_total ≤ 0.
   */
  MixtureLayer(const std::vector<const ScatteringMedium *> &species,
               const std::vector<double> &number_densities,
               double z_min, double z_max);

  /// @brief Exponential free path with the aggregate mean free path `mfp_total`.
  double sample_free_path(luminis::math::Rng &rng) const override;
  double mu_attenuation() const override { return mu_t_total; }
  double mu_scattering() const override { return mu_s_total; }
  double mu_absorption() const override { return mu_a_total; }

  /// @brief Draw one species ∝ μ_s^(i) via a lower_bound on the selection CDF.
  const ScatteringMedium *select_scatter_medium(luminis::math::Rng &rng) const override;
};

// ══════════════════════════════════════════════════════════════════════════════
//  Sample — ordered collection of contiguous layers
// ══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Ordered, contiguous stack of scattering layers forming a physical sample.
 *
 * Layers are stored sorted by `z_min`. The sample enforces:
 *   - The first layer starts at z = 0.
 *   - Each subsequent layer starts exactly where the previous one ends
 *     (no gaps, no overlaps).
 *   - The last layer may extend to z = +∞ (semi-infinite) or to a finite z_max.
 *
 * The sample also holds the **host medium refractive index** (`refractive_index`)
 * which is shared by all layers. The photon velocity
 * (`light_speed = 1.0 / refractive_index`) is derived from this value.
 *
 * Interface z-coordinates (boundaries between adjacent layers) are stored in
 * a sorted vector for efficient O(log N) lookups.
 *
 * ## Example
 * ```cpp
 * Sample sample(1.33);  // water host medium
 * sample.add_layer(&medium1, 0.0, 5.0);               // z ∈ [0, 5)
 * sample.add_layer(&medium2, 5.0, INFINITY);           // z ∈ [5, ∞)
 * ```
 */
struct Sample {
  std::vector<std::unique_ptr<Layer>> layers; ///< Contiguous sorted layers (polymorphic, owned).
  std::vector<double>      interfaces;    ///< Sorted z-coordinates of layer boundaries (excluding 0 and top z_max).

  double refractive_index{1.0};           ///< Real part of the host medium refractive index.
  double light_speed{1.0};                ///< Phase speed of light in the host medium (= 1 / n_medium in natural units).

  /**
   * @brief Construct a Sample with the given host medium refractive index.
   *
   * @param n_medium  Refractive index of the host medium (solvent).
   *                  Defaults to 1.0 (vacuum/air). The photon velocity is
   *                  computed as `1.0 / n_medium`.
   */
  explicit Sample(double n_medium = 1.0);

  // The layers are owned via std::unique_ptr<Layer>, so the Sample is move-only.
  // Copy is explicitly deleted: libc++'s std::is_copy_constructible reports a
  // vector<unique_ptr<…>> as copy-constructible, which would otherwise make
  // pybind11 select its (ill-formed) copy path for Sample* members.
  Sample(const Sample &) = delete;
  Sample &operator=(const Sample &) = delete;
  Sample(Sample &&) = default;
  Sample &operator=(Sample &&) = default;

  /**
   * @brief Return the phase speed of light in the host medium.
   * @return Speed (= 1 / n_medium in natural units).
   */
  double light_speed_in_medium() const;

  /**
   * @brief Add a new layer to the top of the sample.
   *
   * The layer must start exactly where the previous layer ends (or at z = 0
   * if the sample is empty). `z_max` must be greater than `z_min`.
   *
   * @param medium  Non-owning pointer to the scattering medium for this layer.
   * @param z_min   Lower z-boundary [mm].
   * @param z_max   Upper z-boundary [mm] (use INFINITY for unbounded top).
   *
   * @throws std::invalid_argument if the layer is not contiguous with the sample.
   */
  void add_layer(const ScatteringMedium *medium, double z_min, double z_max);

  /**
   * @brief Add a mixture layer (several co-located species) to the top.
   *
   * Builds a `MixtureLayer` from the species and their number densities. Like
   * `add_layer`, the new layer must be contiguous with the current top.
   *
   * @param species           Co-located scattering media (must outlive the sample).
   * @param number_densities  Number density n_i [1/mm³] for each species.
   * @param z_min             Lower z-boundary [mm].
   * @param z_max             Upper z-boundary [mm] (use INFINITY for unbounded top).
   *
   * @throws std::invalid_argument if not contiguous, on size mismatch, or μ_s_total ≤ 0.
   */
  void add_mixture_layer(const std::vector<const ScatteringMedium *> &species,
                         const std::vector<double> &number_densities,
                         double z_min, double z_max);

  /**
   * @brief Return the number of layers in the sample.
   * @return Layer count.
   */
  std::size_t size() const;

  /**
   * @brief Access a layer by index.
   *
   * @param index  Zero-based layer index.
   * @return       Const reference to the Layer.
   */
  const Layer& get_layer(std::size_t index) const;

  /**
   * @brief Find the layer containing a given z-coordinate.
   *
   * Uses binary search on the interfaces vector for O(log N) lookup.
   * Returns `nullptr` if z is outside the sample (z < 0 or z >= top z_max).
   *
   * @param z  z-coordinate [mm].
   * @return   Pointer to the containing Layer, or nullptr.
   */
  const Layer* get_layer_at(double z) const;

  /**
   * @brief Find the layer index containing a given z-coordinate.
   *
   * Returns the zero-based index into `layers`, or `size()` if z is outside.
   *
   * @param z  z-coordinate [mm].
   * @return   Layer index, or `size()` if outside.
   */
  std::size_t get_layer_index_at(double z) const;

  /**
   * @brief Find the z-coordinate of the next interface between two z-values.
   *
   * Given a photon at `z_current` moving toward `z_target`, returns the
   * z-coordinate of the first layer interface crossed, or `std::nullopt`
   * if no interface lies strictly between `z_current` and `z_target`.
   *
   * @param z_current  Current z-position [mm].
   * @param z_target   Target z-position [mm] (may be < or > z_current).
   * @return           z of the next interface, or `std::nullopt`.
   */
  std::optional<double> find_next_interface(double z_current, double z_target) const;

  /**
   * @brief Test whether a 3D position lies inside the sample volume.
   *
   * A position is inside if:
   *   - `z >= 0`
   *   - `z < z_max` of the last layer (or any z if the last layer is infinite)
   *   - No coordinate is NaN or infinite (x, y may be NaN/Inf → outside)
   *
   * @param position  3D position to test [mm].
   * @return          `true` if inside the sample volume.
   */
  bool is_inside(const Vec3 &position) const;

  /**
   * @brief Return the z_max of the topmost layer.
   *
   * May be `+∞` if the top layer is unbounded.
   *
   * @return Maximum z [mm].
   */
  double z_top() const;
};

} // namespace luminis::core
