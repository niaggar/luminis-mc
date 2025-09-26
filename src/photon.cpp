#include <luminis/core/photon.hpp>

namespace luminis::core {

    Photon::Photon(Vec3 p, Vec3 d, double wl) : pos(p), dir(normalize(d)), wavelength_nm(wl) {}

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
