#pragma once
#include "luminis/math.hpp"

namespace luminis {

struct Photon {
  Vec3 pos{0, 0, 0};
  Vec3 dir{1, 0, 0}; // must be normalized
  double wavelength_nm{532.0};
  bool alive{true};

  Photon() = default;
  Photon(Vec3 p, Vec3 d, double wl)
      : pos(p), dir(normalize(d)), wavelength_nm(wl) {}
  void move(double s) { pos = pos + dir * s; }
};

} // namespace luminis
