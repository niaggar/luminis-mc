#pragma once
#include <array>

namespace luminis {

struct Photon {
  std::array<double, 3> position;
  std::array<double, 3> direction;
  double wavelength;

  Photon(std::array<double, 3> pos, std::array<double, 3> dir, double wl);

  void move(double distance);
};

} // namespace luminis
