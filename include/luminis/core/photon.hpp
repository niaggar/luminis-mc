#pragma once
#include <luminis/math/vec.hpp>
#include <sys/types.h>

using namespace luminis::math;

namespace luminis::core {

struct Photon {
  Vec3 pos{0, 0, 0};
  Vec3 dir{1, 0, 0};
  bool alive{true};
  uint events{0};
  double opticalpath{0.0};
  double wavelength_nm;

  Vec2 polarization{1, 0};


  Photon() = default;
  Photon(Vec3 p, Vec3 d, double wl)
      : pos(p), dir(normalize(d)), wavelength_nm(wl) {}
  void move(double s) { pos = pos + dir * s; }
};

} // namespace luminis::core
