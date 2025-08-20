#pragma once
#include <luminis/math/vec.hpp>

using namespace luminis::math;

namespace luminis::core {

// A simple plane detector at z = z0 with circular aperture radius R.
// If a photon crosses the plane (moving +z) within radius R, we count a hit.
struct PlaneDetector {
  double z0{100.0};
  double radius{10.0};
  std::size_t hits{0};

  explicit PlaneDetector(double z_plane, double r)
      : z0(z_plane), radius(r), hits(0) {}

  // Check if a segment from p to p+dir*s crosses z=z0 forward and inside radius
  // Returns true if hit registered.
  bool intersect_and_record(const Vec3 &p, const Vec3 &dir, double s) {
    // If dir.z is zero or negative, cannot cross forward plane
    if (dir[2] <= 0.0)
      return false;

    const double t = (z0 - p[2]) / dir[2]; // distance along ray to reach plane
    if (t < 0.0 || t > s)
      return false; // not within this step

    // position at intersection
    const double x = p[0] + dir[0] * t;
    const double y = p[1] + dir[1] * t;

    if (x * x + y * y <= radius * radius) {
      ++hits;
      return true;
    }
    return false;
  }
};

} // namespace luminis::core
