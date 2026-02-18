#include <luminis/math/utils.hpp>

namespace luminis::math
{
  // Evento de scattering en el estilo de tu forward:
  // E_plane = R * E_in
  // E_out_un = S * E_plane
  // F = ||E_out_un||^2
  // E_out = E_out_un / sqrt(F)   (consistente con tu muestreo)
  CVec2 scatter_event(const CMatrix &S, const CMatrix &R_in_to_plane, const CVec2 &E_in)
  {
    CVec2 E_plane = apply2(R_in_to_plane, E_in);
    CVec2 E_out_un = apply2(S, E_plane);
    double F = norm2(E_out_un);
    if (F < 1e-30)
    {
      return CVec2{std::complex<double>(0, 0), std::complex<double>(0, 0)};
    }
    double inv = 1.0 / std::sqrt(F);
    E_out_un.m *= inv;
    E_out_un.n *= inv;
    return E_out_un; // en la base "plane_out"
  }

  double clamp_pm1(double x)
  {
    if (x > 1.0)
      return 1.0;
    if (x < -1.0)
      return -1.0;
    return x;
  }

  Vec3 row_vec3(const Matrix &P, int r)
  {
    return Vec3(P(r, 0), P(r, 1), P(r, 2));
  }

  Vec3 safe_unit(Vec3 v, Vec3 fallback, double eps)
  {
    double n2 = dot(v, v);
    if (n2 < eps)
    {
      double f2 = dot(fallback, fallback);
      if (f2 < eps)
        return Vec3(1, 0, 0); // último recurso
      return fallback * (1.0 / std::sqrt(f2));
    }
    return v * (1.0 / std::sqrt(n2));
  }

  // Rotación 2x2 que transforma componentes de (m_from,n_from) -> (m_to,n_to)
  CMatrix rot2(const Vec3 &m_to, const Vec3 &n_to, const Vec3 &m_from, const Vec3 &n_from)
  {
    CMatrix R(2, 2);
    R(0, 0) = std::complex<double>(dot(m_to, m_from), 0.0);
    R(0, 1) = std::complex<double>(dot(m_to, n_from), 0.0);
    R(1, 0) = std::complex<double>(dot(n_to, m_from), 0.0);
    R(1, 1) = std::complex<double>(dot(n_to, n_from), 0.0);
    return R;
  }

  CVec2 apply2(const CMatrix &A, const CVec2 &v)
  {
    CVec2 out;
    out.m = A(0, 0) * v.m + A(0, 1) * v.n;
    out.n = A(1, 0) * v.m + A(1, 1) * v.n;
    return out;
  }

  double norm2(const CVec2 &v)
  {
    return std::norm(v.m) + std::norm(v.n);
  }

  double solid_angle_bin(int i_theta, double dtheta, double dphi)
  {
    const double th0 = i_theta * dtheta;
    const double th1 = (i_theta + 1) * dtheta;
    const double dOmega = dphi * (std::cos(th0) - std::cos(th1)); // >0
    return dOmega;
  }
}