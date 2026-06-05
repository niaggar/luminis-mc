#include <luminis/math/vec.hpp>
#include <stdexcept>

namespace luminis::math {

// Maximum number of output elements handled on the stack (no heap allocation).
// All matrices multiplied in the transport/CBS hot paths are 2×2 or 3×3, so a
// 4×4 (16-element) scratch buffer covers every case without allocating.
static constexpr uint LUMINIS_MATMUL_STACK = 16;

double dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3 &a, const Vec3 &b) {
  Vec3 result;
  result.x = a.y * b.z - a.z * b.y;
  result.y = a.z * b.x - a.x * b.z;
  result.z = a.x * b.y - a.y * b.x;

  double norm_factor = norm(result);
  if (norm_factor > 0.0) {
    double inv_norm = 1.0 / norm_factor;
    result = result * inv_norm;
  }

  return result;
}

double norm(const Vec3 &v) {
  return std::sqrt(dot(v, v));
}

void matcmul(const CMatrix &A, const CMatrix &B, CMatrix &C) {
  if (A.cols != B.rows || A.rows != C.rows || B.cols != C.cols) {
    throw std::invalid_argument("CMatrix multiplication dimension mismatch");
  }

  const uint n = A.rows, p = B.cols, kdim = A.cols;
  const uint out = n * p;

  // Fast path: accumulate the full product into a stack buffer first (so the
  // output may safely alias A or B), then copy into C. No heap allocation.
  // Accumulation order is identical to the previous implementation, so results
  // are bit-for-bit unchanged.
  if (out <= LUMINIS_MATMUL_STACK) {
    std::complex<double> tmp[LUMINIS_MATMUL_STACK];
    for (uint i = 0; i < n; ++i) {
      for (uint j = 0; j < p; ++j) {
        std::complex<double> acc(0.0, 0.0);
        for (uint k = 0; k < kdim; ++k) {
          acc += A.data[i * A.cols + k] * B.data[k * B.cols + j];
        }
        tmp[i * p + j] = acc;
      }
    }
    for (uint idx = 0; idx < out; ++idx) {
      C.data[idx] = tmp[idx];
    }
    return;
  }

  CMatrix C_temp(n, p);
  for (uint i = 0; i < n; ++i) {
    for (uint j = 0; j < p; ++j) {
      std::complex<double> acc(0.0, 0.0);
      for (uint k = 0; k < kdim; ++k) {
        acc += A.data[i * A.cols + k] * B.data[k * B.cols + j];
      }
      C_temp.data[i * p + j] = acc;
    }
  }
  C.data.swap(C_temp.data);
}

void matmul(const Matrix &A, const Matrix &B, Matrix &C) {
  if (A.cols != B.rows || A.rows != C.rows || B.cols != C.cols) {
    throw std::invalid_argument("Matrix multiplication dimension mismatch");
  }

  const uint n = A.rows, p = B.cols, kdim = A.cols;
  const uint out = n * p;

  if (out <= LUMINIS_MATMUL_STACK) {
    double tmp[LUMINIS_MATMUL_STACK];
    for (uint i = 0; i < n; ++i) {
      for (uint j = 0; j < p; ++j) {
        double acc = 0.0;
        for (uint k = 0; k < kdim; ++k) {
          acc += A.data[i * A.cols + k] * B.data[k * B.cols + j];
        }
        tmp[i * p + j] = acc;
      }
    }
    for (uint idx = 0; idx < out; ++idx) {
      C.data[idx] = tmp[idx];
    }
    return;
  }

  Matrix C_temp(n, p);
  for (uint i = 0; i < n; ++i) {
    for (uint j = 0; j < p; ++j) {
      double acc = 0.0;
      for (uint k = 0; k < kdim; ++k) {
        acc += A.data[i * A.cols + k] * B.data[k * B.cols + j];
      }
      C_temp.data[i * p + j] = acc;
    }
  }
  C.data.swap(C_temp.data);
}

void matcmulscalar(const double &scalar, CMatrix &A) {
  for (uint i = 0; i < A.rows; ++i) {
    for (uint j = 0; j < A.cols; ++j) {
      A(i, j) *= scalar;
    }
  }
}

double calculate_rotation_angle(const Vec3& u, const Vec3& v) {
    // Unsigned angle between u and v via the dot-product definition.
    // A null vector yields an undefined angle; return 0 (no rotation).
    double norm_u = std::sqrt(dot(u, u));
    double norm_v = std::sqrt(dot(v, v));
    if (norm_u < 1e-12 || norm_v < 1e-12) {
        return 0.0;
    }

    // Clamp the cosine to [-1, 1] to guard acos against NaN from rounding.
    double cos_val = dot(u, v) / (norm_u * norm_v);
    if (cos_val > 1.0) cos_val = 1.0;
    if (cos_val < -1.0) cos_val = -1.0;

    return std::acos(cos_val);
}

}
