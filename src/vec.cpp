#include <luminis/math/vec.hpp>

namespace luminis::math {

double dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(const Vec3 &v) {
  return std::sqrt(dot(v, v));
}

void matmul(const CMatrix &A, const CMatrix &B, CMatrix &C) {
  if (A.cols != B.rows || A.rows != C.rows || B.cols != C.cols) {
    LLOG_ERROR("CMatrix multiplication dimension mismatch");
    std::exit(EXIT_FAILURE);
  }

  for (uint i = 0; i < A.rows; ++i) {
    for (uint j = 0; j < B.cols; ++j) {
      C(i, j) = std::complex<double>(0.0, 0.0);
      for (uint k = 0; k < A.cols; ++k) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

void matmulscalar(const double &scalar, CMatrix &A) {
  for (uint i = 0; i < A.rows; ++i) {
    for (uint j = 0; j < A.cols; ++j) {
      A(i, j) *= scalar;
    }
  }
}

}
