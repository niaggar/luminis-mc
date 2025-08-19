#pragma once
#include <array>
#include <cmath>

namespace luminis {
using Vec3 = std::array<double, 3>;

inline Vec3 operator+(const Vec3 &a, const Vec3 &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
inline Vec3 operator-(const Vec3 &a, const Vec3 &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline Vec3 operator*(const Vec3 &a, double s) {
  return {a[0] * s, a[1] * s, a[2] * s};
}
inline Vec3 operator*(double s, const Vec3 &a) { return a * s; }

inline double dot(const Vec3 &a, const Vec3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline double norm2(const Vec3 &a) { return dot(a, a); }
inline double norm(const Vec3 &a) { return std::sqrt(norm2(a)); }
inline Vec3 normalize(const Vec3 &a) {
  const double n = norm(a);
  return (n > 0) ? (a * (1.0 / n)) : Vec3{0, 0, 0};
}

inline Vec3 from_spherical(double theta, double phi) {
  const double s = std::sin(theta);
  return {s * std::cos(phi), s * std::sin(phi), std::cos(theta)};
}
} // namespace luminis
