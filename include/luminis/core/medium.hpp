/**
 * @file medium.hpp
 * @brief Scattering medium hierarchy for Monte Carlo photon transport.
 *
 * Defines three tiers of scattering media:
 *
 * - **ScatteringMedium** (abstract base): holds the optical coefficients
 *   (μ_a, μ_s, μ_t), a pointer to a PhaseFunction, and provides sampling
 *   interfaces for free path, scattering angle θ, azimuthal angle φ, the
 *   conditional φ sampling that accounts for polarization state, and the
 *   2×2 amplitude scattering matrix J(θ,φ,k).
 *
 * - **RGDMedium**: Rayleigh-Gans-Debye (RGD) approximation.  Uses an
 *   analytical form-factor and a constant exponential free-path distribution.
 *   Suitable for dilute suspensions of small particles (size parameter x ≪ 1).
 *
 * - **MieMedium**: full Mie scattering via MIEV0.  The amplitude functions
 *   S1(θ) and S2(θ) are precomputed into `DataTable` objects at construction
 *   time; subsequent calls to `scattering_matrix` are O(1) table look-ups.
 *
 * @note The amplitude scattering matrix returned by every concrete medium is in
 *       the (p, s) basis:
 *       @code
 *         | s2   0  |
 *         |  0  s1  |
 *       @endcode
 *       where s2 acts on the parallel (p) component and s1 on the perpendicular
 *       (s) component of the electric field.
 *
 * @note The global refractive index of the host medium (solvent) is stored in
 *       the `Sample` class, not in individual scattering media. All layers
 *       share the same host medium; only the particle properties differ.
 *
 * @see luminis/sample/phase.hpp   — PhaseFunction interface
 * @see luminis/sample/table.hpp   — DataTable for tabulated S1/S2
 * @see luminis/mie/dmiev.h        — MIEV0 Mie solver
 * @see luminis/core/sample.hpp    — Sample (layered container with shared n_medium)
 */

#pragma once
#include <complex>
#include <luminis/math/rng.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/sample/phase.hpp>
#include <luminis/sample/meanfreepath.hpp>

using namespace luminis::math;
using namespace luminis::sample;

namespace luminis::core {

// ══════════════════════════════════════════════════════════════════════════════
//  Abstract base scattering medium
// ══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Abstract scattering medium defining the optical transport interface.
 *
 * Stores the bulk optical coefficients and delegates angle/path sampling to
 * a concrete PhaseFunction.  Derived classes must implement:
 *   - `sample_free_path()`  — the free-path length distribution
 *   - `scattering_matrix()` — the 2×2 amplitude scattering matrix J(θ,φ,k)
 *
 * @note The host medium refractive index and photon velocity are managed by
 *       the `Sample` class. `ScatteringMedium` only describes particle-level
 *       scattering and absorption properties.
 */
struct ScatteringMedium {
  const PhaseFunction *phase_function{nullptr}; ///< Pointer to the phase-function sampler (not owned)

  double mu_absorption{0.0};  ///< Absorption coefficient μ_a [1/mm]
  double mu_scattering{0.0};  ///< Scattering coefficient μ_s [1/mm]
  double mu_attenuation{0.0}; ///< Total attenuation coefficient μ_t = μ_a + μ_s [1/mm]

  double n_particle;     ///< Real refractive index of the scatterer
  double n_medium;       ///< Real refractive index of the surrounding medium (used for contrast ratio)

  double wavelength;       ///< Vacuum wavelength of light [mm]
  double k;                ///< Wave number in the medium [1/mm]

  virtual ~ScatteringMedium() = default;

  /**
   * @brief Construct a scattering medium from its bulk optical coefficients.
   *
   * @param phase_func     Pointer to the phase-function sampler.
   *                       Must remain valid for the lifetime of this object.
   */
  ScatteringMedium(PhaseFunction *phase_func);

  // ── Sampling interface ─────────────────────────────────────────────────────

  /**
   * @brief Sample a free path length from the medium's path-length distribution.
   * @param rng  Random number generator.
   * @return     Free path length [mm].
   */
  virtual double sample_free_path(Rng &rng) const = 0;

  /**
   * @brief Sample an azimuthal scattering angle φ unconditionally.
   *
   * Delegates to `phase_function->sample_phi()`.  Terminates the program with
   * an error message if no phase function has been set.
   *
   * @param rng  Random number generator.
   * @return     Azimuthal angle φ ∈ [0, 2π) [rad].
   */
  virtual double sample_azimuthal_angle(Rng &rng) const;

  /**
   * @brief Sample φ conditioned on the current polarization state.
   *
   * Used for polarization-sensitive (CBS) transport.  Delegates to
   * `phase_function->sample_phi_conditional()`.  Terminates with an error if
   * no phase function is set.
   *
   * @param rng    Random number generator.
   * @param S      2×2 amplitude scattering matrix J(θ) for the current event.
   * @param E      Current Jones vector of the photon.
   * @param theta  Polar scattering angle θ [rad].
   * @return       Azimuthal angle φ ∈ [0, 2π) [rad].
   */
  virtual double sample_conditional_azimuthal_angle(Rng &rng, CMatrix& S, CVec2& E, double theta) const;

  /**
   * @brief Sample a polar scattering angle θ from the phase function.
   *
   * Delegates to `phase_function->sample_theta()`.  Terminates with an error
   * if no phase function is set.
   *
   * @param rng  Random number generator.
   * @return     Polar scattering angle θ ∈ [0, π] [rad].
   */
  virtual double sample_scattering_angle(Rng &rng) const;

  /**
   * @brief Return the 2×2 amplitude scattering matrix J(θ, φ, k).
   *
   * The matrix is in the (p, s) basis:
   * @code
   *   | s2   0  |
   *   |  0  s1  |
   * @endcode
   *
   * @param theta  Polar scattering angle θ [rad].
   * @param phi    Azimuthal scattering angle φ [rad].
   * @return       2×2 complex amplitude scattering matrix.
   */
  virtual CMatrix scattering_matrix(const double theta, const double phi) const = 0;

  virtual double scattering_efficiency() const = 0;

  virtual double scattering_cross_section() const = 0;

  void set_scattering_coefficient(double mu_s);

  void set_absorption_coefficient(double mu_a);
};

// ══════════════════════════════════════════════════════════════════════════════
//  Rayleigh-Gans-Debye (RGD) medium
// ══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Scattering medium using the Rayleigh-Gans-Debye (RGD) approximation.
 *
 * Valid for dilute suspensions of small, optically soft particles
 * (size parameter x = 2π r/λ ≪ 1 and |m - 1| ≪ 1).
 * Free paths are exponentially distributed with mean `mean_free_path`.
 * The amplitude scattering matrix is computed analytically from the particle
 * volume, form factor F(θ, k, r), and the contrast index (m - 1).
 */
struct RGDMedium : public ScatteringMedium {
  double mean_free_path; ///< Mean free path l = 1/μ_s [mm]
  double radius;         ///< Particle radius [mm]

  /**
   * @brief Construct a RGDMedium.
   *
   * @param phase_func  Phase-function sampler (borrowing pointer).
   * @param r           Particle radius [mm].
   * @param n_particle  Refractive index of the particle.
   * @param n_medium    Refractive index of the surrounding medium.
   * @param wavelength  Vacuum wavelength of light [mm].
   */
  RGDMedium(PhaseFunction *phase_func, double r, double n_particle, double n_medium, double wavelength);

  /**
   * @brief Sample an exponentially-distributed free path.
   * @param rng  Random number generator.
   * @return     Free path l = -l_mean · ln(U) [mm].
   */
  double sample_free_path(Rng &rng) const override;

  /**
   * @brief Compute the RGD amplitude scattering matrix.
   *
   * Uses the Rayleigh-Gans-Debye formula:
   * @code
   *   s2 = -i k³ (m-1) V F(θ,k,r) cos(θ) / (2π)
   *   s1 = -i k³ (m-1) V F(θ,k,r)         / (2π)
   * @endcode
   * where V = (4π/3) r³ is the particle volume and F is the form factor.
   *
   * @param theta  Polar scattering angle θ [rad].
   * @param phi    Azimuthal scattering angle φ [rad] (unused in RGD).
   * @return       2×2 complex amplitude scattering matrix.
   */
  CMatrix scattering_matrix(const double theta, const double phi) const override;

  double scattering_efficiency() const override;

  double scattering_cross_section() const override;

  void set_mean_free_path(double mfp);
};

// ══════════════════════════════════════════════════════════════════════════════
//  Mie-theory medium (MIEV0 solver with precomputed tables)
// ══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Scattering medium using full Mie theory via the MIEV0 solver.
 *
 * At construction the Mie amplitude functions S1(θ) and S2(θ) are precomputed
 * over a uniform grid of `n_samples` angles in [0, π] and stored in
 * `DataTable` objects for O(1) look-up during transport.
 *
 * Valid for spherical particles of arbitrary size parameter x = 2π r/λ.
 */
struct MieMedium : public ScatteringMedium {
  double mean_free_path;          ///< Mean free path l = 1/μ_s [mm]
  double radius;                  ///< Particle radius [mm]
  double wavelength;              ///< Vacuum wavelength of light [mm]
  std::complex<double> m;         ///< Relative refractive index m = n_particle / n_medium (real part only)

  DataTable S1_table; ///< Precomputed look-up table for the Mie amplitude function S1(θ)
  DataTable S2_table; ///< Precomputed look-up table for the Mie amplitude function S2(θ)

  /**
   * @brief Construct a MieMedium and precompute the S1/S2 look-up tables.
   *
   * Calls `precompute_scattering_tables()` internally with 1000 angle samples.
   *
   * @param phase_func  Phase-function sampler (borrowing pointer).
   * @param r           Particle radius [mm].
   * @param n_particle  Refractive index of the particle.
   * @param n_medium    Refractive index of the surrounding medium.
   * @param wavelength  Vacuum wavelength [mm].
   */
  MieMedium(PhaseFunction *phase_func, double r, double n_particle, double n_medium, double wavelength);

  /**
   * @brief Sample an exponentially-distributed free path.
   * @param rng  Random number generator.
   * @return     Free path l = -l_mean · ln(U) [mm].
   */
  double sample_free_path(Rng &rng) const override;

  /**
   * @brief Return the Mie amplitude scattering matrix via table look-up.
   *
   * Interpolates S1(θ) and S2(θ) from the precomputed tables.  The result
   * is equivalent to the full Mie amplitude scattering matrix evaluated at θ.
   *
   * @param theta  Polar scattering angle θ [rad].
   * @param phi    Azimuthal scattering angle φ [rad] (unused; symmetry is handled upstream).
   * @return       2×2 complex amplitude scattering matrix.
   */
  CMatrix scattering_matrix(const double theta, const double phi) const override;

  /**
   * @brief Precompute S1 and S2 Mie amplitude functions over [0, π].
   *
   * Allocates temporary arrays, calls the MIEV0 Fortran solver (`miev()`),
   * then populates `S1_table` and `S2_table` with the results.
   * Invoked automatically by the constructor.
   *
   * @param wavelength      Vacuum wavelength [mm].
   * @param size_parameter  Mie size parameter x = 2π r / λ.
   * @param n_samples       Number of uniformly-spaced angle samples in [0, π].
   */
  void precompute_scattering_tables(double wavelength, double size_parameter, std::size_t n_samples = 1000);

  double scattering_efficiency() const override;

  double scattering_cross_section() const override;

  void set_mean_free_path(double mfp);
};

} // namespace luminis::core
