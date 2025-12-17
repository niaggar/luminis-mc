#pragma once
#include <complex>
#include <luminis/math/vec.hpp>
#include <sys/types.h>
#include <array>

using namespace luminis::math;

namespace luminis::core {

struct Photon {
  Vec3 prev_pos{0, 0, 0};
  Vec3 pos{0, 0, 0};
  Vec3 dir{Z_UNIT_VEC3};
  Vec3 m{X_UNIT_VEC3};
  Vec3 n{Y_UNIT_VEC3};

  uint events{0};
  double penetration_depth{0.0};
  bool alive{true};
  double wavelength_nm{0.0};
  double k{0.0};
  double opticalpath{0.0};
  double launch_time{0.0};
  double velocity{299792458e-6}; // Speed of light in medium [mm/ns]
  double weight{1.0};

  bool polarized{true};
  CVec2 polarization{std::complex<double>(1, 0), std::complex<double>(0, 0)};


  // CBS related
  Vec3 s_0{0, 0, 0}; // Incident direction (s_in)
  Vec3 s_1{0, 0, 0}; // Direction after 1st scatter

  Vec3 s_n{0, 0, 0}; // Direction after current step (s_n)
  Vec3 s_n1{0, 0, 0}; // Direction before current step (s_n-1)

  Vec3 r_0{0, 0, 0}; // Position first scatter
  Vec3 r_n{0, 0, 0}; // Position last scatter

  CMatrix matrix_T{2, 2}; // Total Jones matrix
  CMatrix matrix_T_buffer{2, 2}; // Buffer for Jones matrix updates


  Photon() = default;
  Photon(const Vec3 &p, const Vec3 &d, const Vec3 &m, const Vec3 &n, const double wl);

  void set_polarization(CVec2 pol);
  std::array<double, 4> get_stokes_parameters() const;
};

struct PhotonRecord {
  double velocity{299792458e-6}; // Speed of light in medium [mm/ns]
  double wavelength_nm{0.0};
  double k{0.0};

  uint events{0};
  double penetration_depth{0.0};
  double launch_time{0.0};
  double arrival_time{0.0};
  double opticalpath{0.0};

  double weight{0.0};
  Vec3 position{0.0, 0.0, 0.0};
  Vec3 direction{0.0, 0.0, 1.0};
  Vec3 m{1.0, 0.0, 0.0};
  Vec3 n{0.0, 1.0, 0.0};
  CVec2 polarization{std::complex<double>(1, 0), std::complex<double>(0, 0)};
};

} // namespace luminis::core
