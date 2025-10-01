#include <luminis/core/photon.hpp>

namespace luminis::core {

    Photon::Photon(const Vec3 &p, const Vec3 &d, const Vec3 &m, const Vec3 &n, const double wl) {
      this->pos[0] = p[0];
      this->pos[1] = p[1];
      this->pos[2] = p[2];

      this->dir[0] = d[0];
      this->dir[1] = d[1];
      this->dir[2] = d[2];

      this->m[0] = m[0];
      this->m[1] = m[1];
      this->m[2] = m[2];

      this->n[0] = n[0];
      this->n[1] = n[1];
      this->n[2] = n[2];

      this->wavelength_nm = wl;
      this->k = 2.0 * M_PI / wavelength_nm;
    }

    void Photon::set_polarization(const CVec2 &pol) {
      this->polarized = true;
      this->polarization[0] = pol[0];
      this->polarization[1] = pol[1];
    }

    std::array<double, 4> Photon::get_stokes_parameters() const {
        if (!polarized) {
            return {1.0, 0.0, 0.0, 0.0};
        }

        const auto &Ex = polarization[0];
        const auto &Ey = polarization[1];

        const double I = std::norm(Ex) + std::norm(Ey);
        const double Q = std::norm(Ex) - std::norm(Ey);
        const double U = 2.0 * std::real(Ex * std::conj(Ey));
        const double V = -2.0 * std::imag(Ex * std::conj(Ey));

        return {I, Q, U, V};
    }

} // namespace luminis::core
