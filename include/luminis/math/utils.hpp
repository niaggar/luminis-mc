#pragma once
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <luminis/math/vec.hpp>

namespace luminis::math
{
  // Evento de scattering en el estilo de tu forward:
  // E_plane = R * E_in
  // E_out_un = S * E_plane
  // F = ||E_out_un||^2
  // E_out = E_out_un / sqrt(F)   (consistente con tu muestreo)
  CVec2 scatter_event(const CMatrix &S, const CMatrix &R_in_to_plane, const CVec2 &E_in);

  double clamp_pm1(double x);

  Vec3 row_vec3(const Matrix &P, int r);

  Vec3 safe_unit(Vec3 v, Vec3 fallback, double eps = 1e-20);

  // Rotación 2x2 que transforma componentes de (m_from,n_from) -> (m_to,n_to)
  CMatrix rot2(const Vec3 &m_to, const Vec3 &n_to, const Vec3 &m_from, const Vec3 &n_from);

  CVec2 apply2(const CMatrix &A, const CVec2 &v);

  double norm2(const CVec2 &v);

  double solid_angle_bin(int i_theta, double dtheta, double dphi);
}