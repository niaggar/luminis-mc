#include <luminis/math/vec.hpp>

namespace luminis::math {

double dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return Vec3{
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  };
}

double norm(const Vec3 &v) {
  return std::sqrt(dot(v, v));
}

void matmul(const CMatrix &A, const CMatrix &B, CMatrix &C) {
  if (A.cols != B.rows || A.rows != C.rows || B.cols != C.cols) {
    LLOG_ERROR("CMatrix multiplication dimension mismatch");
    std::exit(EXIT_FAILURE);
  }

  CMatrix C_temp(A.rows, B.cols);

  for (uint i = 0; i < A.rows; ++i) {
    for (uint j = 0; j < B.cols; ++j) {
      C_temp(i, j) = std::complex<double>(0.0, 0.0);
      for (uint k = 0; k < A.cols; ++k) {
        C_temp(i, j) += A(i, k) * B(k, j);
      }
    }
  }
  C.data.swap(C_temp.data);
}

void matmulscalar(const double &scalar, CMatrix &A) {
  for (uint i = 0; i < A.rows; ++i) {
    for (uint j = 0; j < A.cols; ++j) {
      A(i, j) *= scalar;
    }
  }
}

double calculate_rotation_angle(const Vec3& n_from, const Vec3& n_to) {
  const double cos_theta = dot(n_from, n_to);
  double norm_from = norm(n_from);
  double norm_to = norm(n_to);

  const double cos_theta_clamped = std::clamp(cos_theta / (norm_from * norm_to), -1.0, 1.0);
  return std::acos(cos_theta_clamped);
}

}
