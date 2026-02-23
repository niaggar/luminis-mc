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
#include <optional>
#include <vector>
#include <luminis/core/medium.hpp>

namespace luminis::core {

// ══════════════════════════════════════════════════════════════════════════════
//  SampleLayer — one slab of the layered sample
// ══════════════════════════════════════════════════════════════════════════════

/**
 * @brief A single layer in a layered sample.
 *
 * Associates a non-owning pointer to a `ScatteringMedium` with a half-open
 * z-range `[z_min, z_max)`. For the topmost layer, `z_max` may be `+∞`
 * (`std::numeric_limits<double>::infinity()`), indicating an unbounded
 * semi-infinite slab.
 */
struct SampleLayer {
  const ScatteringMedium *medium{nullptr}; ///< Non-owning pointer to the scattering medium for this layer.
  double z_min{0.0};                       ///< Lower z-boundary of the layer [mm] (inclusive).
  double z_max{0.0};                       ///< Upper z-boundary of the layer [mm] (exclusive, or +∞).

  /**
   * @brief Construct a SampleLayer.
   *
   * @param medium  Pointer to the scattering medium (must remain valid for the sample lifetime).
   * @param z_min   Lower z-boundary [mm].
   * @param z_max   Upper z-boundary [mm] (use INFINITY for unbounded top).
   */
  SampleLayer(const ScatteringMedium *medium, double z_min, double z_max);

  /**
   * @brief Test whether a z-coordinate lies within this layer.
   *
   * @param z  z-coordinate [mm].
   * @return   `true` if `z_min <= z < z_max`.
   */
  bool contains(double z) const;

  /**
   * @brief Return the thickness of the layer.
   *
   * Returns `+∞` if `z_max` is infinite.
   *
   * @return Thickness [mm].
   */
  double thickness() const;
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
  std::vector<SampleLayer> layers;        ///< Contiguous sorted layers.
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
   * @brief Return the number of layers in the sample.
   * @return Layer count.
   */
  std::size_t size() const;

  /**
   * @brief Access a layer by index.
   *
   * @param index  Zero-based layer index.
   * @return       Const reference to the SampleLayer.
   */
  const SampleLayer& get_layer(std::size_t index) const;

  /**
   * @brief Find the layer containing a given z-coordinate.
   *
   * Uses binary search on the interfaces vector for O(log N) lookup.
   * Returns `nullptr` if z is outside the sample (z < 0 or z >= top z_max).
   *
   * @param z  z-coordinate [mm].
   * @return   Pointer to the containing SampleLayer, or nullptr.
   */
  const SampleLayer* get_layer_at(double z) const;

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
