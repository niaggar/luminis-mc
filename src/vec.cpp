#include <luminis/math/vec.hpp>

namespace luminis::math {

double dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(const Vec3 &v) {
  return std::sqrt(dot(v, v));
}

CMatrix matmul(const CMatrix &A, const CMatrix &B) {
    CMatrix C(2, 2);
    C(0,0) = A(0,0)*B(0,0) + A(0,1)*B(1,0);
    C(0,1) = A(0,0)*B(0,1) + A(0,1)*B(1,1);
    C(1,0) = A(1,0)*B(0,0) + A(1,1)*B(1,0);
    C(1,1) = A(1,0)*B(0,1) + A(1,1)*B(1,1);
    return C;
}

}
