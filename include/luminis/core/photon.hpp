#pragma once
#include <complex>
#include <luminis/math/vec.hpp>
#include <sys/types.h>
#include <array>

using namespace luminis::math;

namespace luminis::core
{

  struct Photon
  {
    Vec3 prev_pos{0, 0, 0};
    Vec3 pos{0, 0, 0};
    Vec3 detected_pos{0, 0, 0};

    uint events{0};
    double penetration_depth{0.0};
    bool alive{true};
    double wavelength_nm{0.0};
    double k{0.0};
    double opticalpath{0.0};
    double launch_time{0.0};
    double velocity{1.0}; // Speed of light in medium [mm/ns]
    double weight{1.0};

    Matrix P_local = Matrix(3, 3); // Local scattering plane basis

    bool polarized{true};
    CVec2 polarization{std::complex<double>(1, 0), std::complex<double>(0, 0)};
    CVec2 polarization_reverse{std::complex<double>(1, 0), std::complex<double>(0, 0)};

    // CBS related
    bool coherent_path_calculated{false};
    CVec2 initial_polarization{std::complex<double>(1, 0), std::complex<double>(0, 0)};

    Matrix P0 = Matrix(3, 3);   // Initial scattering plane basis
    Matrix P1 = Matrix(3, 3);   // Scattering plane basis after first scatter
    Matrix Pn2 = Matrix(3, 3);  // Scattering plane basis after second last scatte
    Matrix Pn1 = Matrix(3, 3);  // Scattering plane basis after last scatter
    Matrix Pn = Matrix(3, 3);   // Scattering plane basis after last scatter

    Vec3 r_0{0, 0, 0}; // Position first scatter
    Vec3 r_n{0, 0, 0}; // Position last scatter

    CMatrix matrix_T = CMatrix::identity(2); // Total Jones matrix
    CMatrix matrix_T_buffer = CMatrix::identity(2); // Buffer for Jones matrix updates
    bool has_T_prev{false}; // Flag to indicate if previous T matrix is available

    Photon() = default;
    Photon(const Vec3 &p, const Vec3 &d, const Vec3 &m, const Vec3 &n, const double wl);

    void set_polarization(CVec2 pol);
    std::array<double, 4> get_stokes_parameters() const;
  };

  struct PhotonRecord
  {
    uint events{0};
    double penetration_depth{0.0};
    double launch_time{0.0};
    double arrival_time{0.0};
    double opticalpath{0.0};
    double weight{0.0};
    double k{0.0};

    Vec3 position_first_scattering{0.0, 0.0, 0.0};
    Vec3 position_last_scattering{0.0, 0.0, 0.0};
    Vec3 position_detector{0.0, 0.0, 0.0};

    Vec3 direction{0.0, 0.0, 1.0};
    Vec3 m{1.0, 0.0, 0.0};
    Vec3 n{0.0, 1.0, 0.0};

    CVec2 polarization_forward{std::complex<double>(1, 0), std::complex<double>(0, 0)};
    CVec2 polarization_reverse{std::complex<double>(0, 0), std::complex<double>(0, 0)};
  };

} // namespace luminis::core
