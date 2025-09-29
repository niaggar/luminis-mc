#pragma once
#include <cstddef>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <vector>

using namespace luminis::math;

namespace luminis::core {

struct Detector {
  std::size_t hits{0};
  Vec3 origin{0, 0, 0};
  Vec3 normal{0, 0, 1};
  std::vector<Photon> recorded_photons;

  Detector() = default;
  Detector(const Vec3 &o, const Vec3 &n);

  void record_hit(Photon &photon);
};

} // namespace luminis::core
