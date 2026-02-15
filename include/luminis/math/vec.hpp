#pragma once
#include "luminis/log/logger.hpp"
#include <cmath>
#include <complex>
#include <cstdlib>
#include <sys/types.h>
#include <vector>

namespace luminis::math {

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

struct Vec2 {
  double x{0.0};
  double y{0.0};

  Vec2() = default;
  Vec2(double x, double y) : x(x), y(y) {}
};

struct CVec2 {
  std::complex<double> m{0.0, 0.0};
  std::complex<double> n{0.0, 0.0};

  CVec2() = default;
  CVec2(std::complex<double> m, std::complex<double> n) : n(n), m(m) {}
};

struct Matrix {
  uint rows;
  uint cols;
  std::vector<double> data;

  Matrix() = default;

  Matrix(uint rows, uint cols) : rows(rows), cols(cols) {
    data.resize(rows * cols, 0.0);
  }

  double& operator()(uint i, uint j) {
    if (i >= rows || j >= cols) {
      LLOG_ERROR("Matrix index out of range");
      std::exit(EXIT_FAILURE);
    }
    return data[i * cols + j];
  }

  const double& operator()(uint i, uint j) const {
    if (i >= rows || j >= cols) {
      LLOG_ERROR("Matrix index out of range");
      std::exit(EXIT_FAILURE);
    }
    return data[i * cols + j];
  }
};

struct CMatrix {
  uint rows;
  uint cols;
  std::vector<std::complex<double>> data;

  CMatrix() = default;

  CMatrix(uint rows, uint cols) : rows(rows), cols(cols) {
    data.resize(rows * cols, std::complex<double>(0.0, 0.0));
  }

  std::complex<double>& operator()(uint i, uint j) {
    if (i >= rows || j >= cols) {
      LLOG_ERROR("CMatrix index out of range");
      std::exit(EXIT_FAILURE);
    }
    return data[i * cols + j];
  }

  const std::complex<double>& operator()(uint i, uint j) const {
    if (i >= rows || j >= cols) {
      LLOG_ERROR("CMatrix index out of range");
      std::exit(EXIT_FAILURE);
    }
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

double dot(const Vec3 &a, const Vec3 &b);
Vec3 cross(const Vec3 &a, const Vec3 &b);
double norm(const Vec3 &v);
void matcmul(const CMatrix &A, const CMatrix &B, CMatrix &C);
void matcmulscalar(const double &scalar, CMatrix &A);
void matmul(const Matrix &A, const Matrix &B, Matrix &C);
double calculate_rotation_angle(const Vec3& n_from, const Vec3& n_to);

} // namespace luminis::math
