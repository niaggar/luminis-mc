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
  Vec3 n_polarization{1, 0, 0};
  Vec3 m_polarization{0, 1, 0};
  std::vector<Photon> recorded_photons;

  Detector() = default;
  Detector(const Vec3 o, const Vec3 normal, const Vec3 n, const Vec3 m);

  void record_hit(Photon &photon);


};

} // namespace luminis::core
