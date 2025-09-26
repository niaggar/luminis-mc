#pragma once
#include <luminis/math/vec.hpp>

using namespace luminis::math;

namespace luminis::core {

struct Detector {
  std::size_t hits{0};

  void record_hit(const Vec3 &position, const Vec3 &direction);



};

} // namespace luminis::core
