#pragma once
#include <luminis/math/vec.hpp>
#include <sys/types.h>
#include <complex>

using namespace luminis::math;


namespace luminis::core {

struct Photon {
  Vec3 prev_pos{0, 0, 0};
  Vec3 pos{0, 0, 0};
  Vec3 dir{0, 0, 1};
  Vec3 m{1, 0, 0};
  Vec3 n{0, 1, 0};

  uint events{0};
  bool alive{true};
  double wavelength_nm;
  double k{2.0 * M_PI / wavelength_nm};
  double opticalpath{0.0};
  double weight{1.0};

  bool polarized{false};
  CVec2 polarization{std::complex<double>(1, 0), std::complex<double>(0, 0)};

  Photon() = default;
  Photon(Vec3 p, Vec3 d, double wl);

  std::array<double, 4> get_stokes_parameters() const;
};

} // namespace luminis::core
