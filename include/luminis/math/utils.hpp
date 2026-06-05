/**
 * @file utils.hpp
 * @brief Small Jones-calculus and geometry helpers for polarized transport.
 *
 * Free functions used throughout the scattering kernel and CBS reconstruction:
 * applying a scattering event to a Jones vector, building 2×2 rotation matrices
 * between local frames, and miscellaneous numeric helpers.
 */

#pragma once
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <luminis/math/vec.hpp>

namespace luminis::math
{
  /// @brief Apply one scattering event to a Jones vector and renormalize.
  /// @details Computes E_out = S · (R · E_in), normalized so that |E_out|² = 1
  ///          (returns the zero vector if the result is numerically null).
  /// @param S             2×2 amplitude scattering matrix.
  /// @param R_in_to_plane Rotation from the incoming frame to the scattering plane.
  /// @param E_in          Incoming Jones vector.
  /// @return Normalized outgoing Jones vector in the outgoing-plane basis.
  CVec2 scatter_event(const CMatrix &S, const CMatrix &R_in_to_plane, const CVec2 &E_in);

  /// @brief Clamp `x` to the closed interval [-1, 1] (e.g. before acos).
  double clamp_pm1(double x);

  /// @brief Extract row `r` of a matrix as a Vec3 (expects 3 columns).
  Vec3 row_vec3(const Matrix &P, int r);

  /// @brief Normalize `v`, falling back to `fallback` (then to +x) if `v` is ~zero.
  /// @param eps Squared-norm threshold below which a vector is treated as null.
  Vec3 safe_unit(Vec3 v, Vec3 fallback, double eps = 1e-20);

  /// @brief Build the 2×2 rotation mapping components from (m_from, n_from) to (m_to, n_to).
  CMatrix rot2(const Vec3 &m_to, const Vec3 &n_to, const Vec3 &m_from, const Vec3 &n_from);

  /// @brief Apply a 2×2 complex matrix to a Jones vector: out = A · v.
  CVec2 apply2(const CMatrix &A, const CVec2 &v);

  /// @brief Squared magnitude |v|² = |v.m|² + |v.n|² of a Jones vector.
  double norm2(const CVec2 &v);

  /// @brief Solid angle [sr] subtended by the i-th polar bin: dφ · (cos θ₀ − cos θ₁).
  /// @param i_theta Polar bin index.
  /// @param dtheta  Polar bin width [rad].
  /// @param dphi    Azimuthal bin width [rad].
  double solid_angle_bin(int i_theta, double dtheta, double dphi);
}