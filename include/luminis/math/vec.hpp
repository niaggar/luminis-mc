#pragma once
#include "luminis/log/logger.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <sys/types.h>
#include <utility>
#include <vector>

namespace luminis::math {

/**
 * @brief Small-buffer-optimized contiguous storage for Matrix/CMatrix.
 *
 * Matrices in the per-photon hot path are always tiny (2×2 or 3×3), while the
 * sensor grids are large (N_x×N_y, N_theta×N_phi, …).  This container keeps up
 * to @c SBO_CAP elements inline (no heap allocation) and transparently falls
 * back to a heap `std::vector` for anything larger.  The public surface mirrors
 * the subset of `std::vector` the codebase relies on (`resize`, `operator[]`,
 * `data()`, `size()`, `swap()`), so it is a drop-in replacement.
 *
 * Storage is always contiguous, so `data()` remains valid for the NumPy buffer
 * protocol bindings.  There are no self-referential pointers, so the default
 * copy/move/assignment are correct.
 */
template <typename T>
struct SmallStorage {
  static constexpr std::size_t SBO_CAP = 16; // covers every 2×2/3×3/4×4 hot-path matrix

  std::array<T, SBO_CAP> small_{};
  std::vector<T> heap_{};
  std::size_t size_{0};
  bool on_heap_{false};

  void resize(std::size_t n, T value = T{}) {
    if (n > SBO_CAP) {
      heap_.assign(n, value);
      on_heap_ = true;
    } else {
      for (std::size_t i = 0; i < n; ++i)
        small_[i] = value;
      on_heap_ = false;
    }
    size_ = n;
  }

  // Hot-path element access: `on_heap_` is a perfectly predictable branch (always
  // false for the tiny transport matrices) and the inline path avoids the heap
  // pointer indirection a std::vector would incur on every access.
  T &operator[](std::size_t i) { return on_heap_ ? heap_[i] : small_[i]; }
  const T &operator[](std::size_t i) const { return on_heap_ ? heap_[i] : small_[i]; }

  T *data() { return on_heap_ ? heap_.data() : small_.data(); }
  const T *data() const { return on_heap_ ? heap_.data() : small_.data(); }

  std::size_t size() const { return size_; }

  void swap(SmallStorage &o) {
    std::swap(small_, o.small_);
    heap_.swap(o.heap_);
    std::swap(size_, o.size_);
    std::swap(on_heap_, o.on_heap_);
  }
};

/// @brief 3D vector with component-wise arithmetic (positions and directions).
struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};

  Vec3() = default;
  Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

  Vec3 operator+(const Vec3& other) const {
    return Vec3{x + other.x, y + other.y, z + other.z};
  }
  Vec3 operator-(const Vec3& other) const {
    return Vec3{x - other.x, y - other.y, z - other.z};
  }
  Vec3 operator*(double scalar) const {
    return Vec3{x * scalar, y * scalar, z * scalar};
  }
  Vec3 operator/(double scalar) const {
    return Vec3{x / scalar, y / scalar, z / scalar};
  }
};

const Vec3 ZERO_VEC3{0.0, 0.0, 0.0};
const Vec3 X_UNIT_VEC3{1.0, 0.0, 0.0};
const Vec3 Y_UNIT_VEC3{0.0, 1.0, 0.0};
const Vec3 Z_UNIT_VEC3{0.0, 0.0, 1.0};

/// @brief 2D real vector.
struct Vec2 {
  double x{0.0};
  double y{0.0};

  Vec2() = default;
  Vec2(double x, double y) : x(x), y(y) {}
};

/// @brief Jones vector: complex field amplitudes along the local (m, n) axes.
struct CVec2 {
  std::complex<double> m{0.0, 0.0}; ///< Amplitude along the m-axis.
  std::complex<double> n{0.0, 0.0}; ///< Amplitude along the n-axis.

  CVec2() = default;
  CVec2(std::complex<double> m, std::complex<double> n) : n(n), m(m) {}
};

/// @brief Dense row-major real matrix backed by SmallStorage (SBO for tiny matrices).
struct Matrix {
  uint rows;
  uint cols;
  SmallStorage<double> data;

  Matrix() = default;

  Matrix(uint rows, uint cols) : rows(rows), cols(cols) {
    data.resize(rows * cols, 0.0);
  }

  // Bounds checks are kept as assert() so they are active in Debug builds but
  // compiled out under NDEBUG (Release) — element access sits in the per-event
  // hot loop and must not carry a branch + library call.
  double& operator()(uint i, uint j) {
    assert(i < rows && j < cols && "Matrix index out of range");
    return data[i * cols + j];
  }

  const double& operator()(uint i, uint j) const {
    assert(i < rows && j < cols && "Matrix index out of range");
    return data[i * cols + j];
  }
};

/// @brief Dense row-major complex matrix backed by SmallStorage (SBO for tiny matrices).
struct CMatrix {
  uint rows;
  uint cols;
  SmallStorage<std::complex<double>> data;

  CMatrix() = default;

  CMatrix(uint rows, uint cols) : rows(rows), cols(cols) {
    data.resize(rows * cols, std::complex<double>(0.0, 0.0));
  }

  std::complex<double>& operator()(uint i, uint j) {
    assert(i < rows && j < cols && "CMatrix index out of range");
    return data[i * cols + j];
  }

  const std::complex<double>& operator()(uint i, uint j) const {
    assert(i < rows && j < cols && "CMatrix index out of range");
    return data[i * cols + j];
  }

  static CMatrix identity(uint size) {
    CMatrix I(size, size);
    for (uint i = 0; i < size; ++i) {
      I(i, i) = std::complex<double>(1.0, 0.0);
    }
    return I;
  }
};

/// @brief Dot product a · b.
double dot(const Vec3 &a, const Vec3 &b);
/// @brief Cross product a × b.
Vec3 cross(const Vec3 &a, const Vec3 &b);
/// @brief Euclidean norm |v|.
double norm(const Vec3 &v);
/// @brief Complex matrix product C = A · B (C written in place, no allocation).
void matcmul(const CMatrix &A, const CMatrix &B, CMatrix &C);
/// @brief Scale a complex matrix in place: A *= scalar.
void matcmulscalar(const double &scalar, CMatrix &A);
/// @brief Real matrix product C = A · B (C written in place, no allocation).
void matmul(const Matrix &A, const Matrix &B, Matrix &C);
/// @brief Signed rotation angle [rad] taking direction n_from to n_to.
double calculate_rotation_angle(const Vec3& n_from, const Vec3& n_to);

} // namespace luminis::math
