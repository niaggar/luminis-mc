#pragma once
#include <cmath>
#include <complex>
#include <sys/types.h>

namespace luminis::math {

struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};

  Vec3() = default;
  Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
};

struct Vec2 {
  double x{0.0};
  double y{0.0};

  Vec2() = default;
  Vec2(double x, double y) : x(x), y(y) {}
};

struct CVec2 {
  std::complex<double> n{0.0, 0.0};
  std::complex<double> m{0.0, 0.0};

  CVec2() = default;
  CVec2(std::complex<double> m, std::complex<double> n) : n(n), m(m) {}
};

struct Matrix {
  const uint rows;
  const uint cols;
  double* data;

  Matrix(uint rows, uint cols) : rows(rows), cols(cols) {
    data = new double[rows * cols]();
  }

  ~Matrix() {
    delete[] data;
  }

  double& operator()(uint i, uint j) {
    return data[i * cols + j];
  }

  const double& operator()(uint i, uint j) const {
    return data[i * cols + j];
  }
};

struct CMatrix {
  const uint rows;
  const uint cols;
  std::complex<double>* data;

  CMatrix(uint rows, uint cols) : rows(rows), cols(cols) {
    data = new std::complex<double>[rows * cols]();
  }

  ~CMatrix() {
    delete[] data;
  }

  std::complex<double>& operator()(uint i, uint j) {
    return data[i * cols + j];
  }

  const std::complex<double>& operator()(uint i, uint j) const {
    return data[i * cols + j];
  }
};

double dot(const Vec3 &a, const Vec3 &b);
double norm(const Vec3 &v);

} // namespace luminis::math
