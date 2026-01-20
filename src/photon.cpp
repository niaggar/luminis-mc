#include "luminis/math/vec.hpp"
#include <luminis/core/photon.hpp>

namespace luminis::core {

    Photon::Photon(const Vec3 &p, const Vec3 &d, const Vec3 &m, const Vec3 &n, const double wl) {
      this->pos.x = p.x;
      this->pos.y = p.y;
      this->pos.z = p.z;

      this->prev_pos.x = p.x;
      this->prev_pos.y = p.y;
      this->prev_pos.z = p.z;

      this->dir.x = d.x;
      this->dir.y = d.y;
      this->dir.z = d.z;

      this->m.x = m.x;
      this->m.y = m.y;
      this->m.z = m.z;

      this->n.x = n.x;
      this->n.y = n.y;
      this->n.z = n.z;

      this->wavelength_nm = wl;
      this->k = 2.0 * M_PI / wavelength_nm;
    }

    void Photon::set_polarization(CVec2 pol) {
      this->polarized = true;
      this->polarization.m = pol.m;
      this->polarization.n = pol.n;
    }

    std::array<double, 4> Photon::get_stokes_parameters() const {
        if (!polarized) {
            return {1.0, 0.0, 0.0, 0.0};
        }

        const auto &Ex = polarization.m;
        const auto &Ey = polarization.n;

        const double I = std::norm(Ex) + std::norm(Ey);
        const double Q = std::norm(Ex) - std::norm(Ey);
        const double U = 2.0 * std::real(Ex * std::conj(Ey));
        const double V = -2.0 * std::imag(Ex * std::conj(Ey));

        return {I, Q, U, V};
    }

} // namespace luminis::core
