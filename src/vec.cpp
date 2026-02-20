#include <luminis/math/vec.hpp>

namespace luminis::math {

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

void matmul(const Matrix &A, const Matrix &B, Matrix &C) {
  if (A.cols != B.rows || A.rows != C.rows || B.cols != C.cols) {
    LLOG_ERROR("Matrix multiplication dimension mismatch");
    std::exit(EXIT_FAILURE);
  }

  Matrix C_temp(A.rows, B.cols);

  for (uint i = 0; i < A.rows; ++i) {
    for (uint j = 0; j < B.cols; ++j) {
      C_temp(i, j) = 0.0;
      for (uint k = 0; k < A.cols; ++k) {
        C_temp(i, j) += A(i, k) * B(k, j);
      }
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
  // 1. Verificar vectores nulos (evitar división por cero en normalización interna si la hubiera)
    double norm_u = std::sqrt(dot(u, u));
    double norm_v = std::sqrt(dot(v, v));
    
    if (norm_u < 1e-12 || norm_v < 1e-12) {
        return 0.0; // Si un vector es nulo, el ángulo es indefinido (asumimos 0 rotación)
    }

    // 2. Calcular coseno
    double cos_val = dot(u, v) / (norm_u * norm_v);

    // 3. CLAMPING (Vital para evitar NaN en acos)
    if (cos_val > 1.0) cos_val = 1.0;
    if (cos_val < -1.0) cos_val = -1.0;

    // 4. Determinar signo (usualmente se necesita un tercer vector de referencia para el signo en 0..2pi, 
    // pero para rotaciones de matrices de scattering a veces acos basta si es simétrico. 
    // Nota: Tu función original 'calculate_rotation_angle' probablemente maneja el signo usando cross product.
    // Si es así, asegúrate de que esa lógica también tenga guards).
    
    // Asumiendo que tu función original 'calculate_rotation_angle' hace algo más complejo para el signo:
    // Lo mejor es NO usar la función original si los vectores son peligrosos.
    
    return std::acos(cos_val);
}

}
