#include <luminis/math/vec.hpp>

namespace luminis::math {

double dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(const Vec3 &v) {
  return std::sqrt(dot(v, v));
}

}
