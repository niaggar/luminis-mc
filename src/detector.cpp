#include <cmath>
#include <complex>
#include <luminis/core/detector.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/log/logger.hpp>
#include <vector>

namespace luminis::core
{

  // Detector implementation
  Detector::Detector(double z)
  {
    origin = {0, 0, z};
    normal = Z_UNIT_VEC3;
    backward_normal = Z_UNIT_VEC3 * -1;
    m_polarization = X_UNIT_VEC3;
    n_polarization = Y_UNIT_VEC3;
  }

  std::unique_ptr<Detector> Detector::clone() const
  {
    auto det = std::make_unique<Detector>(origin.z);
    return det;
  }

  void Detector::merge_from(const Detector &other)
  {
    hits += other.hits;
    recorded_photons.insert(recorded_photons.end(),
                            other.recorded_photons.begin(),
                            other.recorded_photons.end());
  }

  bool Detector::is_hit_by(const Photon &photon) const
  {
    const bool valid = validate_detection_conditions(photon);
    if (!valid)
      return false;

    const Vec3 xn = photon.prev_pos;
    const Vec3 xf = photon.pos;
    const Vec3 d{
        xf.x - xn.x,
        xf.y - xn.y,
        xf.z - xn.z};

    const double denom = dot(d, normal);

    // Plane and line are parallel
    if (std::abs(denom) < 1e-6)
      return false;

    const double t = dot({origin.x - xn.x,
                          origin.y - xn.y,
                          origin.z - xn.z},
                         normal) /
                     denom;

    // Intersection point is not between xn and xf
    if (t < 0 || t > 1)
      return false;

    return true;
  }

  void Detector::record_hit(Photon &photon)
  {
    const bool valid = validate_detection_conditions(photon);
    if (!valid)
      return;

    const Vec3 xn = photon.prev_pos;
    const Vec3 xf = photon.pos;
    const Vec3 d{
        xf.x - xn.x,
        xf.y - xn.y,
        xf.z - xn.z};

    const double denom = dot(d, normal);

    // Plane and line are parallel
    if (std::abs(denom) < 1e-6)
      return;

    const double t = dot({origin.x - xn.x,
                          origin.y - xn.y,
                          origin.z - xn.z},
                         normal) /
                     denom;

    // Intersection point is not between xn and xf
    if (t < 0 || t > 1)
      return;

    const Vec3 hit_point{
        xn.x + t * d.x,
        xn.y + t * d.y,
        xn.z + t * d.z};
    const double correction_distance = luminis::math::norm({hit_point.x - xf.x,
                                                            hit_point.y - xf.y,
                                                            hit_point.z - xf.z});
    if (correction_distance > 0)
    {
      photon.opticalpath -= correction_distance;
    }

    hits += 1;
    photon.pos = hit_point;

    PhotonRecord photon_rec{};
    photon_rec.events = photon.events;
    photon_rec.penetration_depth = photon.penetration_depth;
    photon_rec.launch_time = photon.launch_time;
    photon_rec.arrival_time = photon.launch_time + (photon.opticalpath / photon.velocity);
    photon_rec.opticalpath = photon.opticalpath;
    photon_rec.weight = photon.weight;
    photon_rec.k = photon.k;
    photon_rec.position_detector = photon.pos;
    photon_rec.position_first_scattering = photon.r_0;
    photon_rec.position_last_scattering = photon.r_n;
    photon_rec.direction = photon.dir;
    photon_rec.m = photon.m;
    photon_rec.n = photon.n;
    photon_rec.polarization_forward = photon.polarization;
    photon_rec.polarization_reverse = CVec2{std::complex<double>(0, 0), std::complex<double>(0, 0)};

    recorded_photons.push_back(photon_rec);
  }

  // AngleDetector implementation
  AngleDetector::AngleDetector(double z, int n_theta, int n_phi)
      : Detector(z),
        N_theta(n_theta),
        N_phi(n_phi),
        dtheta((M_PI / 2.0) / n_theta),
        dphi((2.0 * M_PI) / n_phi),
        E_x(n_theta, n_phi),
        E_y(n_theta, n_phi),
        E_z(n_theta, n_phi)
  {
  }

  void AngleDetector::record_hit(Photon &photon)
  {
    const bool valid = validate_detection_conditions(photon);
    if (!valid)
      return;

    // Get angles
    const Vec3 u = photon.dir;
    const double costtheta = -1 * u.z;
    const double theta = std::acos(costtheta);
    double phi = std::atan2(u.y, u.x);
    if (phi < 0)
      phi += 2.0 * M_PI;

    // Determine bins
    const int itheta = std::min(static_cast<int>(std::floor((theta / (M_PI / 2.0)) * N_theta)), N_theta - 1);
    const int iphi = std::min(static_cast<int>(std::floor((phi / (2.0 * M_PI)) * N_phi)), N_phi - 1);

    // Compute local field contribution
    std::complex<double> phase = std::exp(std::complex<double>(0, photon.k * photon.opticalpath));
    std::complex<double> Em_local_photon = photon.polarization.m * phase * std::sqrt(photon.weight);
    std::complex<double> En_local_photon = photon.polarization.n * phase * std::sqrt(photon.weight);

    // Accumulate field contributions
    E_x(itheta, iphi) += Em_local_photon * photon.m.x + En_local_photon * photon.n.x;
    E_y(itheta, iphi) += Em_local_photon * photon.m.y + En_local_photon * photon.n.y;
    E_z(itheta, iphi) += Em_local_photon * photon.m.z + En_local_photon * photon.n.z;
  }

  std::unique_ptr<Detector> AngleDetector::clone() const
  {
    auto det = std::make_unique<AngleDetector>(origin.z, N_theta, N_phi);
    return det;
  }

  void AngleDetector::merge_from(const Detector &other)
  {
    const AngleDetector &other_speckle = dynamic_cast<const AngleDetector &>(other);
    hits += other_speckle.hits;

    for (int itheta = 0; itheta < N_theta; ++itheta)
    {
      for (int iphi = 0; iphi < N_phi; ++iphi)
      {
        E_x(itheta, iphi) += other_speckle.E_x(itheta, iphi);
        E_y(itheta, iphi) += other_speckle.E_y(itheta, iphi);
        E_z(itheta, iphi) += other_speckle.E_z(itheta, iphi);
      }
    }
  }

  // Detection condition factories
  DetectionCondition make_theta_condition(double min_theta, double max_theta) {
    return [min_theta, max_theta](const Photon &photon) {
      const Vec3 u = photon.dir;
      const double costtheta = -1 * u.z;
      const double theta = std::acos(costtheta);
      return (theta >= min_theta && theta <= max_theta);
    };
  };

  DetectionCondition make_phi_condition(double min_phi, double max_phi) {
    return [min_phi, max_phi](const Photon &photon) {
      const Vec3 u = photon.dir;
      double phi = std::atan2(u.y, u.x);
      if (phi < 0)
        phi += 2.0 * M_PI;
      return (phi >= min_phi && phi <= max_phi);
    };
  };

  DetectionCondition make_position_condition(double min_x, double max_x, double min_y, double max_y) {
    return [min_x, max_x, min_y, max_y](const Photon &photon) {
      const Vec3 pos = photon.pos;
      return (pos.x >= min_x && pos.x <= max_x && pos.y >= min_y && pos.y <= max_y);
    };
  };

  DetectionCondition make_events_condition(uint min_events, uint max_events) {
    return [min_events, max_events](const Photon &photon) {
      return (photon.events >= min_events && photon.events <= max_events);
    };
  };
} // namespace luminis::core
