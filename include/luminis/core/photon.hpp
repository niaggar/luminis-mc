#pragma once
#include <complex>
#include <luminis/math/vec.hpp>
#include <sys/types.h>

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
  double wavelength_nm{0.0};
  double k{0.0};
  double opticalpath{0.0};
  double launch_time{0.0};
  double velocity{299792458e-6}; // Speed of light in medium [mm/ns]
  double weight{1.0};

  bool polarized{true};
  CVec2 polarization{std::complex<double>(1, 0), std::complex<double>(0, 0)};

  Photon() = default;
  Photon(const Vec3 &p, const Vec3 &d, const Vec3 &m, const Vec3 &n, const double wl);

  void set_polarization(std::complex<double> pol1, std::complex<double> pol2);
  std::array<double, 4> get_stokes_parameters() const;
};

} // namespace luminis::core
