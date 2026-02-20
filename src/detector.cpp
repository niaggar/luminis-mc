/**
 * @file detector.cpp
 * @brief Implementation of the sensor/detector system for photon detection.
 *
 * This file contains the implementations for all sensor types used in
 * Monte Carlo photon transport simulations. Sensor types range from simple
 * photon recorders to sophisticated coherent backscattering (CBS) detectors
 * that track forward and reverse electromagnetic field amplitudes.
 *
 * Key algorithms implemented here:
 * - Z-plane intersection for photon hit detection (SensorsGroup::record_hit)
 * - Stokes parameter accumulation from Jones vector fields
 * - Next-event estimation (forced detection) for variance reduction
 * - Three-stage reverse path computation for CBS (coherent_calculation)
 * - CBS geometric phase and coherent/incoherent Stokes decomposition
 *
 * @see detector.hpp for class declarations and detailed API documentation.
 */

#include <cmath>
#include <complex>
#include <luminis/core/detector.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/core/medium.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/math/utils.hpp>
#include <luminis/log/logger.hpp>
#include <vector>
#include <functional>

namespace luminis::core
{
  static inline bool intersect_plane(const Vec3 &r, const Vec3 &v, const Vec3 &p0, const Vec3 &n, double &t_out)
  {
    const double denom = dot(v, n);
    if (std::abs(denom) < 1e-14)
      return false;
    const Vec3 dp = p0 - r;
    const double t = dot(dp, n) / denom;
    if (t <= 0.0)
      return false;
    t_out = t;
    return true;
  }

  // Normalización de fase function implícita en S_matrix:
  // I_norm = ∫_0^π (|S11|^2 + |S22|^2) sinθ dθ
  static double compute_I_norm(const Medium &medium, double k, int n = 2048)
  {
    double acc = 0.0;
    const double dth = M_PI / n;
    for (int i = 0; i < n; i++)
    {
      const double th = (i + 0.5) * dth;
      const CMatrix S = medium.scattering_matrix(th, 0.0, k);
      const double s22 = std::norm(S(0, 0));
      const double s11 = std::norm(S(1, 1));
      acc += (s11 + s22) * std::sin(th);
    }
    return acc * dth;
  }

  // Aplica T = S*R y normaliza usando F (tu misma fórmula)
  static inline CVec2 apply_scatter_normalized(const CMatrix &S, double cos_phi, double sin_phi, const CVec2 &Ein)
  {
    // R(phi)
    CMatrix R(2, 2);
    R(0, 0) = cos_phi;
    R(0, 1) = sin_phi;
    R(1, 0) = -sin_phi;
    R(1, 1) = cos_phi;

    CMatrix T(2, 2);
    matcmul(S, R, T);

    const std::complex<double> Em = Ein.m;
    const std::complex<double> En = Ein.n;

    const double Emm = std::norm(Em);
    const double Enn = std::norm(En);

    const double s22 = std::norm(S(0, 0));
    const double s11 = std::norm(S(1, 1));

    const double c2 = cos_phi * cos_phi;
    const double s2 = sin_phi * sin_phi;

    double F =
        Emm * (s22 * c2 + s11 * s2) +
        Enn * (s22 * s2 + s11 * c2) +
        2.0 * std::real(Em * std::conj(En)) * (s22 - s11) * sin_phi * cos_phi;

    if (F < 1e-300)
    {
      return CVec2{0.0, 0.0};
    }

    const double inv = 1.0 / std::sqrt(F);
    // Eout = inv * T * Ein
    CVec2 Eout;
    Eout.m = inv * (T(0, 0) * Em + T(0, 1) * En);
    Eout.n = inv * (T(1, 0) * Em + T(1, 1) * En);
    return Eout;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // SensorsGroup implementation
  // ═══════════════════════════════════════════════════════════════════════════
  void SensorsGroup::add_detector(std::unique_ptr<Sensor> detector)
  {
    // Assign a unique sequential ID and register in the z-layer lookup map.
    // Sensors at the same z-plane share a map entry for efficient intersection tests.
    u_int new_id = static_cast<u_int>(detectors.size());
    detector->id = new_id;
    double z = detector->origin.z;

    z_layers[z].push_back(detector.get());
    detectors.push_back(std::move(detector));
  }

  bool SensorsGroup::record_hit(Photon &photon, const Medium &medium)
  {
    bool photon_killed = false;

    // Determine the z-range traversed by the photon in this step.
    // Skip if the photon moved purely within a horizontal plane (no z displacement).
    double z1 = photon.prev_pos.z;
    double z2 = photon.pos.z;
    if (std::abs(z2 - z1) < 1e-12)
      return false;

    double z_min = std::min(z1, z2);
    double z_max = std::max(z1, z2);

    // Use the sorted z_layers map to efficiently find only the detector planes
    // that lie within [z_min, z_max]. This avoids iterating over all detectors.
    auto it_start = z_layers.lower_bound(z_min);
    auto it_end = z_layers.upper_bound(z_max);

    for (auto it = it_start; it != it_end; ++it)
    {
      double z_plane = it->first;
      bool crosses = (z_plane >= z_min) && (z_plane <= z_max);

      if (crosses)
      {
        // --- Ray-plane intersection ---
        // Compute the exact intersection point on the z-plane using
        // parametric line equation: hit = prev_pos + t * (pos - prev_pos).
        // All detector planes have normal = (0,0,1).
        const Vec3 xn = photon.prev_pos;
        const Vec3 xf = photon.pos;
        const Vec3 d = xf - xn;
        const Vec3 detector_normal{0, 0, 1};
        const Vec3 detector_origin{0, 0, z_plane};

        const double denom = dot(d, detector_normal);
        const double t = dot(detector_origin - xn, detector_normal) / denom;

        // Compute the hit point and correct the optical path length.
        // The photon may have overshot the plane, so we subtract the
        // extra distance to get the path length at the intersection.
        const Vec3 hit_point = xn + d * t;
        const double correction_distance = luminis::math::norm(hit_point - xf);
        double opticalpath_correction = photon.opticalpath;
        if (correction_distance > 0)
        {
          opticalpath_correction -= correction_distance;
        }

        // Build the interaction info with the corrected propagation phase.
        InteractionInfo info;
        info.intersection_point = hit_point;
        info.phase = std::exp(std::complex<double>(0, photon.k * opticalpath_correction));

        for (Sensor *det : it->second)
        {
          Vec3 direction = {photon.P_local(2, 0), photon.P_local(2, 1), photon.P_local(2, 2)};
          const bool valid_photon = det->check_conditions(hit_point, direction);
          if (valid_photon)
          {
            det->process_hit(photon, info, medium);
            if (det->absorb_photons)
            {
              photon_killed = true;
            }
          }
        }
      }
    }

    return photon_killed;
  }

  void SensorsGroup::run_estimators(const Photon &photon, const Medium &medium)
  {
    for (const auto &detector : detectors)
    {
      if (detector->estimator_enabled)
      {
        detector->process_estimation(photon, medium);
      }
    }
  }

  void SensorsGroup::merge_from(const SensorsGroup &other)
  {
    for (const auto &det_other : other.detectors)
    {
      u_int det_id = det_other->id;
      if (det_id < detectors.size())
      {
        detectors[det_id]->merge_from(*det_other);
      }
      else
      {
        LLOG_WARN("SensorsGroup merge_from: Sensor ID {} not found in destination", det_id);
      }
    }
  }

  std::unique_ptr<SensorsGroup> SensorsGroup::clone() const
  {
    auto multi_det = std::make_unique<SensorsGroup>();
    for (const auto &detector : detectors)
    {
      multi_det->add_detector(detector->clone());
    }
    return multi_det;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Sensor base class implementation
  // ═══════════════════════════════════════════════════════════════════════════

  // Sensor implementation
  Sensor::Sensor(double z, bool absorb, bool estimator)
  {
    origin = {0, 0, z};
    normal = Z_UNIT_VEC3;
    backward_normal = Z_UNIT_VEC3 * -1;
    m_polarization = X_UNIT_VEC3;
    n_polarization = Y_UNIT_VEC3;
    absorb_photons = absorb;

    if (z != 0.0 && estimator)
    {
      LLOG_WARN("Sensor: Estimator mode enabled but z={} is not 0. Estimation is only valid for sensors at z=0. Consider setting z=0 or disabling estimator mode.", z);
      estimator_enabled = false;
    }
    else
    {
      estimator_enabled = estimator;
    }
  }

  void Sensor::set_theta_limit(double min, double max)
  {
    filter_theta_enabled = true;
    filter_theta_min = min;
    filter_theta_max = max;
    double c1 = std::cos(min);
    double c2 = std::cos(max);
    _cache_cos_theta_min = std::min(c1, c2);
    _cache_cos_theta_max = std::max(c1, c2);
  }

  void Sensor::set_phi_limit(double min, double max)
  {
    filter_phi_enabled = true;
    filter_phi_min = min;
    filter_phi_max = max;
  }

  void Sensor::set_position_limit(double x_min, double x_max, double y_min, double y_max)
  {
    filter_position_enabled = true;
    filter_x_min = x_min;
    filter_x_max = x_max;
    filter_y_min = y_min;
    filter_y_max = y_max;
  }

  bool Sensor::check_conditions(const Vec3 &hit_point, const Vec3 &hit_direction) const
  {
    // Apply all enabled filters in sequence (theta, phi, position).
    // Returns false as soon as any filter rejects the photon.
    if (filter_theta_enabled)
    {
      // Use cached cosine bounds to avoid repeated acos() calls.
      // Note: cos is monotonically decreasing, so the bounds are swapped.
      const double costtheta = -1 * hit_direction.z;
      if (costtheta < _cache_cos_theta_min || costtheta > _cache_cos_theta_max)
        return false;
    }

    if (filter_phi_enabled)
    {
      double phi = std::atan2(hit_direction.y, hit_direction.x);
      if (phi < 0)
        phi += 2.0 * M_PI;
      if (phi < filter_phi_min || phi > filter_phi_max)
        return false;
    }

    if (filter_position_enabled)
    {
      if (hit_point.x < filter_x_min || hit_point.x > filter_x_max)
        return false;
      if (hit_point.y < filter_y_min || hit_point.y > filter_y_max)
        return false;
    }

    return true;
  }

  void Sensor::process_estimation(const Photon &photon, const Medium &medium)
  {
    // Default implementation does nothing, can be overridden by specific sensors
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // PhotonRecordSensor implementation
  // ═══════════════════════════════════════════════════════════════════════════
  PhotonRecordSensor::PhotonRecordSensor(double z, bool absorb) : Sensor(z, absorb, false)
  {
  }

  std::unique_ptr<Sensor> PhotonRecordSensor::clone() const
  {
    auto det = std::make_unique<PhotonRecordSensor>(origin.z, absorb_photons);
    det->filter_theta_enabled = filter_theta_enabled;
    det->filter_theta_min = filter_theta_min;
    det->filter_theta_max = filter_theta_max;
    det->_cache_cos_theta_min = _cache_cos_theta_min;
    det->_cache_cos_theta_max = _cache_cos_theta_max;
    det->filter_phi_enabled = filter_phi_enabled;
    det->filter_phi_min = filter_phi_min;
    det->filter_phi_max = filter_phi_max;
    det->filter_position_enabled = filter_position_enabled;
    det->filter_x_min = filter_x_min;
    det->filter_x_max = filter_x_max;
    det->filter_y_min = filter_y_min;
    det->filter_y_max = filter_y_max;
    return det;
  }

  void PhotonRecordSensor::merge_from(const Sensor &other)
  {
    const auto &o = dynamic_cast<const PhotonRecordSensor &>(other);
    hits += o.hits;
    recorded_photons.insert(recorded_photons.end(), o.recorded_photons.begin(), o.recorded_photons.end());
  }

  void PhotonRecordSensor::process_hit(Photon &photon, InteractionInfo &info, const Medium &medium)
  {
    hits += 1;

    PhotonRecord photon_rec{};
    photon_rec.events = photon.events;
    photon_rec.penetration_depth = photon.penetration_depth;
    photon_rec.launch_time = photon.launch_time;
    photon_rec.arrival_time = photon.launch_time + (photon.opticalpath / photon.velocity);
    photon_rec.opticalpath = photon.opticalpath;
    photon_rec.weight = photon.weight;
    photon_rec.k = photon.k;
    photon_rec.position_detector = info.intersection_point;
    photon_rec.position_first_scattering = photon.r_0;
    photon_rec.position_last_scattering = photon.r_n;
    photon_rec.direction = Vec3{photon.P_local(2, 0), photon.P_local(2, 1), photon.P_local(2, 2)};
    photon_rec.m = Vec3{photon.P_local(0, 0), photon.P_local(0, 1), photon.P_local(0, 2)};
    photon_rec.n = Vec3{photon.P_local(1, 0), photon.P_local(1, 1), photon.P_local(1, 2)};
    photon_rec.polarization_forward = photon.polarization;
    photon_rec.polarization_reverse = photon.polarization_reverse;

    recorded_photons.push_back(photon_rec);
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // PlanarFieldSensor implementation
  // ═══════════════════════════════════════════════════════════════════════════
  PlanarFieldSensor::PlanarFieldSensor(double z, double len_x, double len_y, double dx, double dy, bool absorb, bool estimator) : Sensor(z, absorb, estimator)
  {
    this->len_x = len_x;
    this->len_y = len_y;
    this->dx = dx;
    this->dy = dy;

    N_x = static_cast<int>(std::ceil(len_x / dx));
    N_y = static_cast<int>(std::ceil(len_y / dy));

    Ex = CMatrix(N_x, N_y);
    Ey = CMatrix(N_x, N_y);

    const double half_len_x = 0.5 * len_x;
    const double half_len_y = 0.5 * len_y;
    set_position_limit(-half_len_x, half_len_x, -half_len_y, half_len_y);
  }

  std::unique_ptr<Sensor> PlanarFieldSensor::clone() const
  {
    auto det = std::make_unique<PlanarFieldSensor>(origin.z, len_x, len_y, dx, dy, absorb_photons, estimator_enabled);
    det->filter_theta_enabled = filter_theta_enabled;
    det->filter_theta_min = filter_theta_min;
    det->filter_theta_max = filter_theta_max;
    det->_cache_cos_theta_min = _cache_cos_theta_min;
    det->_cache_cos_theta_max = _cache_cos_theta_max;
    det->filter_phi_enabled = filter_phi_enabled;
    det->filter_phi_min = filter_phi_min;
    det->filter_phi_max = filter_phi_max;
    det->filter_position_enabled = filter_position_enabled;
    det->filter_x_min = filter_x_min;
    det->filter_x_max = filter_x_max;
    det->filter_y_min = filter_y_min;
    det->filter_y_max = filter_y_max;
    return det;
  }

  void PlanarFieldSensor::merge_from(const Sensor &other)
  {
    const auto &o = dynamic_cast<const PlanarFieldSensor &>(other);
    hits += o.hits;
    for (int i = 0; i < N_x; ++i)
    {
      for (int j = 0; j < N_y; ++j)
      {
        Ex(i, j) += o.Ex(i, j);
        Ey(i, j) += o.Ey(i, j);
      }
    }
  }

  void PlanarFieldSensor::process_hit(Photon &photon, InteractionInfo &info, const Medium &medium)
  {
    // Map intersection point to grid coordinates (centered at origin).
    const double x = info.intersection_point.x;
    const double y = info.intersection_point.y;
    const int x_idx = static_cast<int>((x + 0.5 * len_x) / dx);
    const int y_idx = static_cast<int>((y + 0.5 * len_y) / dy);

    if (x_idx < 0 || x_idx >= N_x || y_idx < 0 || y_idx >= N_y)
      return;

    // Project the Jones vector from local (m,n) basis to lab (x,y) frame.
    // Weight-corrected amplitude with propagation phase.
    const double w_sqrt = std::sqrt(photon.weight);
    const std::complex<double> Em_local_photon = photon.polarization.m;
    const std::complex<double> En_local_photon = photon.polarization.n;

    Matrix P = photon.P_local;
    Ex(x_idx, y_idx) += (Em_local_photon * P(0, 0) + En_local_photon * P(1, 0)) * info.phase * w_sqrt;
    Ey(x_idx, y_idx) += (Em_local_photon * P(0, 1) + En_local_photon * P(1, 1)) * info.phase * w_sqrt;

    hits += 1;
  }

  void PlanarFieldSensor::process_estimation(const Photon &photon, const Medium &medium)
  {
    const double x = photon.pos.x;
    const double y = photon.pos.y;
    const int x_idx = static_cast<int>((x + 0.5 * len_x) / dx);
    const int y_idx = static_cast<int>((y + 0.5 * len_y) / dy);
    if (x_idx < 0 || x_idx >= N_x || y_idx < 0 || y_idx >= N_y)
      return;

    // Local base of the detector plane
    double ud = 0;
    double vd = 0;
    double wd = 1;

    Matrix Pold = photon.P_local;
    Matrix Q = Matrix(3, 3);
    Matrix A = Matrix(3, 3);

    double mu = Pold(2, 0) * ud + Pold(2, 1) * vd + Pold(2, 2) * wd; // Cosine of angle between photon direction and detector normal
    double nu = sqrt(1 - mu * mu);                                   // Sine of the same angle
    double F = 1;

    std::complex<double> E1old = photon.polarization.m;
    std::complex<double> E2old = photon.polarization.n;
    std::complex<double> Ed1;
    std::complex<double> Ed2;
    Vec3 p;

    if (std::abs(1 - mu) < 1e-11)
    {
      // Photon is on the same direction as the detector normal
      Q = Pold;

      CMatrix Smatrix = medium.scattering_matrix(1.0, 0.0, photon.k);
      double s2 = std::norm(Smatrix(0, 0));
      double s1 = std::norm(Smatrix(1, 1));

      F = (s2 + s1) / 2.0;
      Ed1 = E1old * s2 / sqrt(F);
      Ed2 = E2old * s1 / sqrt(F);
    }
    else if (std::abs(1 + mu) < 1e-11)
    {
      // Photon is on the opposite direction of the detector normal
      Q = Pold;
      Q(1, 0) *= -1;
      Q(1, 1) *= -1;
      Q(1, 2) *= -1;
      Q(2, 0) *= -1;
      Q(2, 1) *= -1;
      Q(2, 2) *= -1;

      CMatrix Smatrix = medium.scattering_matrix(M_PI, 0.0, photon.k);
      double s2 = std::norm(Smatrix(0, 0));
      double s1 = std::norm(Smatrix(1, 1));

      F = (s2 + s1) / 2.0;
      Ed1 = E1old * s2 / sqrt(F);
      Ed2 = E2old * s1 / sqrt(F);
    }
    else
    {
      // Cross product to find the rotation axis and angle
      p = {
          (Pold(2, 1) * wd - Pold(2, 2) * vd) / nu,
          (Pold(2, 2) * ud - Pold(2, 0) * wd) / nu,
          (Pold(2, 0) * vd - Pold(2, 1) * ud) / nu};

      // Dot product to find the cosine and sine of the rotation angle
      double sinphi = -(Pold(0, 0) * p.x + Pold(0, 1) * p.y + Pold(0, 2) * p.z); // m dot p
      double cosphi = Pold(1, 0) * p.x + Pold(1, 1) * p.y + Pold(1, 2) * p.z;    // n dot p

      A(0, 0) = mu * cosphi;
      A(0, 1) = mu * sinphi;
      A(0, 2) = -nu;
      A(1, 0) = -sinphi;
      A(1, 1) = cosphi;
      A(1, 2) = 0;
      A(2, 0) = nu * cosphi;
      A(2, 1) = nu * sinphi;
      A(2, 2) = mu;

      matmul(A, Pold, Q);

      double theta = std::acos(mu);
      CMatrix Smatrix = medium.scattering_matrix(theta, 0, photon.k);
      double s2 = std::norm(Smatrix(0, 0));
      double s1 = std::norm(Smatrix(1, 1));

      double s2sq = std::norm(Smatrix(0, 0));
      double s1sq = std::norm(Smatrix(1, 1));

      double e1sq = std::norm(E1old);
      double e2sq = std::norm(E2old);
      double e12 = (E1old * conj(E2old)).real();

      F = (s2sq * e1sq + s1sq * e2sq) * cosphi * cosphi + (s1sq * e1sq + s2sq * e2sq) * sinphi * sinphi + 2 * (s2sq - s1sq) * e12 * cosphi * sinphi;
      Ed1 = (cosphi * E1old + sinphi * E2old) * s2 / sqrt(F);
      Ed2 = (-sinphi * E1old + cosphi * E2old) * s1 / sqrt(F);
    }

    double deposit = 0.0;
    double z = photon.pos.z;
    double zd = origin.z;
    double weight = photon.weight;
    double csca = 1.0;

    if (photon.events == 0)
      /* ballistic light */
      if (std::abs(1 - mu) < 1e-11)
        deposit = weight * exp(-fabs((z - zd) / wd));
      else
        deposit = 0;
    else
      deposit = weight * F / csca * exp(-fabs((z - zd) / wd));

    double t = photon.launch_time + (photon.opticalpath / photon.velocity);
    double td = t + fabs((z - zd) / wd);

    std::complex<double> phase = std::exp(std::complex<double>(0, photon.k * td));
    std::complex<double> Ex = (Ed1 * Q(0, 0) + Ed2 * Q(1, 0)) * phase;
    std::complex<double> Ey = (Ed1 * Q(0, 1) + Ed2 * Q(1, 1)) * phase;

    this->Ex(x_idx, y_idx) += Ex * sqrt(deposit);
    this->Ey(x_idx, y_idx) += Ey * sqrt(deposit);
    hits += 1;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // PlanarFluenceSensor implementation
  // ═══════════════════════════════════════════════════════════════════════════
  PlanarFluenceSensor::PlanarFluenceSensor(double z, double len_x, double len_y, double len_t, double dx, double dy, double dt, bool absorb, bool estimator) : Sensor(z, absorb, estimator)
  {
    this->len_x = len_x;
    this->len_y = len_y;
    this->len_t = len_t;
    this->dx = dx;
    this->dy = dy;
    this->dt = dt;

    if (dt == 0)
    {
      N_t = 1;
    }
    else
    {
      N_t = static_cast<int>(std::ceil(len_t / dt));
    }

    N_x = static_cast<int>(std::ceil(len_x / dx));
    N_y = static_cast<int>(std::ceil(len_y / dy));

    S0_t.resize(N_t, Matrix(N_x, N_y));
    S1_t.resize(N_t, Matrix(N_x, N_y));
    S2_t.resize(N_t, Matrix(N_x, N_y));
    S3_t.resize(N_t, Matrix(N_x, N_y));

    const double half_len_x = 0.5 * len_x;
    const double half_len_y = 0.5 * len_y;
    set_position_limit(-half_len_x, half_len_x, -half_len_y, half_len_y);
  }

  std::unique_ptr<Sensor> PlanarFluenceSensor::clone() const
  {
    auto det = std::make_unique<PlanarFluenceSensor>(origin.z, len_x, len_y, len_t, dx, dy, dt, absorb_photons, estimator_enabled);
    det->filter_theta_enabled = filter_theta_enabled;
    det->filter_theta_min = filter_theta_min;
    det->filter_theta_max = filter_theta_max;
    det->_cache_cos_theta_min = _cache_cos_theta_min;
    det->_cache_cos_theta_max = _cache_cos_theta_max;
    det->filter_phi_enabled = filter_phi_enabled;
    det->filter_phi_min = filter_phi_min;
    det->filter_phi_max = filter_phi_max;
    det->filter_position_enabled = filter_position_enabled;
    det->filter_x_min = filter_x_min;
    det->filter_x_max = filter_x_max;
    det->filter_y_min = filter_y_min;
    det->filter_y_max = filter_y_max;
    return det;
  }

  void PlanarFluenceSensor::merge_from(const Sensor &other)
  {
    const auto &o = dynamic_cast<const PlanarFluenceSensor &>(other);
    hits += o.hits;
    for (int t = 0; t < N_t; ++t)
    {
      for (int i = 0; i < N_x; ++i)
      {
        for (int j = 0; j < N_y; ++j)
        {
          S0_t[t](i, j) += o.S0_t[t](i, j);
          S1_t[t](i, j) += o.S1_t[t](i, j);
          S2_t[t](i, j) += o.S2_t[t](i, j);
          S3_t[t](i, j) += o.S3_t[t](i, j);
        }
      }
    }
  }

  void PlanarFluenceSensor::process_hit(Photon &photon, InteractionInfo &info, const Medium &medium)
  {
    // --- Time-resolved binning ---
    // If dt==0, all photons go into a single time bin (steady-state mode).
    int t_idx;
    if (dt == 0)
    {
      t_idx = 0;
    }
    else
    {
      double arrival_time = photon.launch_time + (photon.opticalpath / photon.velocity);
      if (arrival_time < 0 || arrival_time >= len_t)
        return;
      t_idx = static_cast<int>(arrival_time / dt);
    }

    // --- Spatial binning (grid centered at origin) ---
    const double x = info.intersection_point.x;
    const double y = info.intersection_point.y;
    const int x_idx = static_cast<int>((x + 0.5 * len_x) / dx);
    const int y_idx = static_cast<int>((y + 0.5 * len_y) / dy);
    if (x_idx < 0 || x_idx >= N_x || y_idx < 0 || y_idx >= N_y)
      return;

    // --- Project Jones vector to lab frame and compute Stokes parameters ---
    // Weight-corrected amplitude with propagation phase included.
    const double w_sqrt = std::sqrt(photon.weight);
    const std::complex<double> Em_local_photon = photon.polarization.m * info.phase * w_sqrt;
    const std::complex<double> En_local_photon = photon.polarization.n * info.phase * w_sqrt;

    // Project local Jones vector to lab frame (x,y).
    // Note: E_det_x is negated to match the convention for the x-component.
    Matrix P = photon.P_local;
    const std::complex<double> E_det_x = (Em_local_photon * P(0, 0) + En_local_photon * P(1, 0)) * -1.0;
    const std::complex<double> E_det_y = (Em_local_photon * P(0, 1) + En_local_photon * P(1, 1));

    // Compute Stokes parameters from the electric field components:
    //   S0 = |Ex|^2 + |Ey|^2  (total intensity)
    //   S1 = |Ex|^2 - |Ey|^2  (linear polarization, 0/90 deg)
    //   S2 = 2 Re(Ex Ey*)     (linear polarization, +/-45 deg)
    //   S3 = 2 Im(Ex Ey*)     (circular polarization)
    const double S0_contribution = std::norm(E_det_x) + std::norm(E_det_y);
    const double S1_contribution = std::norm(E_det_x) - std::norm(E_det_y);
    const double S2_contribution = 2.0 * std::real(E_det_x * std::conj(E_det_y));
    const double S3_contribution = 2.0 * std::imag(E_det_x * std::conj(E_det_y));

    S0_t[t_idx](x_idx, y_idx) += S0_contribution;
    S1_t[t_idx](x_idx, y_idx) += S1_contribution;
    S2_t[t_idx](x_idx, y_idx) += S2_contribution;
    S3_t[t_idx](x_idx, y_idx) += S3_contribution;

    hits += 1;
  }

  void PlanarFluenceSensor::process_estimation(const Photon &photon, const Medium &medium)
  {
    const double x = photon.pos.x;
    const double y = photon.pos.y;
    const int x_idx = static_cast<int>((x + 0.5 * len_x) / dx);
    const int y_idx = static_cast<int>((y + 0.5 * len_y) / dy);
    if (x_idx < 0 || x_idx >= N_x || y_idx < 0 || y_idx >= N_y)
      return;

    // Local base of the detector plane
    double ud = 0;
    double vd = 0;
    double wd = 1;

    Matrix Pold = photon.P_local;
    Matrix Q = Matrix(3, 3);
    Matrix A = Matrix(3, 3);

    double mu = Pold(2, 0) * ud + Pold(2, 1) * vd + Pold(2, 2) * wd; // Cosine of angle between photon direction and detector normal
    double nu = sqrt(1 - mu * mu);                                   // Sine of the same angle
    double F = 1;

    std::complex<double> E1old = photon.polarization.m;
    std::complex<double> E2old = photon.polarization.n;
    std::complex<double> Ed1;
    std::complex<double> Ed2;
    Vec3 p;

    if (std::abs(1 - mu) < 1e-11)
    {
      // Photon is on the same direction as the detector normal
      Q = Pold;

      CMatrix Smatrix = medium.scattering_matrix(1.0, 0.0, photon.k);
      double s2 = std::norm(Smatrix(0, 0));
      double s1 = std::norm(Smatrix(1, 1));

      F = (s2 + s1) / 2.0;
      Ed1 = E1old * s2 / sqrt(F);
      Ed2 = E2old * s1 / sqrt(F);
    }
    else if (std::abs(1 + mu) < 1e-11)
    {
      // Photon is on the opposite direction of the detector normal
      Q = Pold;
      Q(1, 0) *= -1;
      Q(1, 1) *= -1;
      Q(1, 2) *= -1;
      Q(2, 0) *= -1;
      Q(2, 1) *= -1;
      Q(2, 2) *= -1;

      CMatrix Smatrix = medium.scattering_matrix(M_PI, 0.0, photon.k);
      double s2 = std::norm(Smatrix(0, 0));
      double s1 = std::norm(Smatrix(1, 1));

      F = (s2 + s1) / 2.0;
      Ed1 = E1old * s2 / sqrt(F);
      Ed2 = E2old * s1 / sqrt(F);
    }
    else
    {
      // Cross product to find the rotation axis and angle
      p = {
          (Pold(2, 1) * wd - Pold(2, 2) * vd) / nu,
          (Pold(2, 2) * ud - Pold(2, 0) * wd) / nu,
          (Pold(2, 0) * vd - Pold(2, 1) * ud) / nu};

      // Dot product to find the cosine and sine of the rotation angle
      double sinphi = -(Pold(0, 0) * p.x + Pold(0, 1) * p.y + Pold(0, 2) * p.z); // m dot p
      double cosphi = Pold(1, 0) * p.x + Pold(1, 1) * p.y + Pold(1, 2) * p.z;    // n dot p

      A(0, 0) = mu * cosphi;
      A(0, 1) = mu * sinphi;
      A(0, 2) = -nu;
      A(1, 0) = -sinphi;
      A(1, 1) = cosphi;
      A(1, 2) = 0;
      A(2, 0) = nu * cosphi;
      A(2, 1) = nu * sinphi;
      A(2, 2) = mu;

      matmul(A, Pold, Q);

      double theta = std::acos(mu);
      CMatrix Smatrix = medium.scattering_matrix(theta, 0, photon.k);
      double s2 = std::norm(Smatrix(0, 0));
      double s1 = std::norm(Smatrix(1, 1));

      double s2sq = std::norm(Smatrix(0, 0));
      double s1sq = std::norm(Smatrix(1, 1));

      double e1sq = std::norm(E1old);
      double e2sq = std::norm(E2old);
      double e12 = (E1old * conj(E2old)).real();

      F = (s2sq * e1sq + s1sq * e2sq) * cosphi * cosphi + (s1sq * e1sq + s2sq * e2sq) * sinphi * sinphi + 2 * (s2sq - s1sq) * e12 * cosphi * sinphi;
      Ed1 = (cosphi * E1old + sinphi * E2old) * s2 / sqrt(F);
      Ed2 = (-sinphi * E1old + cosphi * E2old) * s1 / sqrt(F);
    }

    double deposit = 0.0;
    double z = photon.pos.z;
    double zd = origin.z;
    double weight = photon.weight;
    double csca = 1.0;

    if (photon.events == 0)
      /* ballistic light */
      if (std::abs(1 - mu) < 1e-11)
        deposit = weight * exp(-fabs((z - zd) / wd));
      else
        deposit = 0;
    else
      deposit = weight * F / csca * exp(-fabs((z - zd) / wd));

    double t = photon.launch_time + (photon.opticalpath / photon.velocity);
    double td = t + fabs((z - zd) / wd);

    std::complex<double> phase = std::exp(std::complex<double>(0, photon.k * td));
    std::complex<double> Ex = (Ed1 * Q(0, 0) + Ed2 * Q(1, 0)) * phase * -1.0;
    std::complex<double> Ey = (Ed1 * Q(0, 1) + Ed2 * Q(1, 1)) * phase;

    const double S0_contribution = std::norm(Ex) + std::norm(Ey);
    const double S1_contribution = std::norm(Ex) - std::norm(Ey);
    const double S2_contribution = 2.0 * std::real(Ex * std::conj(Ey));
    const double S3_contribution = 2.0 * std::imag(Ex * std::conj(Ey));

    int t_idx;
    if (dt == 0)
    {
      t_idx = 0;
    }
    else    {
      double arrival_time = photon.launch_time + (photon.opticalpath / photon.velocity) + fabs((z - zd) / wd);
      if (arrival_time < 0 || arrival_time >= len_t)
        return;
      t_idx = static_cast<int>(arrival_time / dt);
    }

    S0_t[t_idx](x_idx, y_idx) += S0_contribution * deposit;
    S1_t[t_idx](x_idx, y_idx) += S1_contribution * deposit;
    S2_t[t_idx](x_idx, y_idx) += S2_contribution * deposit;
    S3_t[t_idx](x_idx, y_idx) += S3_contribution * deposit;

    hits += 1;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // PlanarCBSSensor implementation
  // ═══════════════════════════════════════════════════════════════════════════
  PlanarCBSSensor::PlanarCBSSensor(double len_x, double len_y, double dx, double dy, bool estimator) : Sensor(0.0, true, estimator)
  {
    this->len_x = len_x;
    this->len_y = len_y;
    this->dx = dx;
    this->dy = dy;

    N_x = static_cast<int>(std::ceil(len_x / dx));
    N_y = static_cast<int>(std::ceil(len_y / dy));

    S0 = Matrix(N_x, N_y);
    S1 = Matrix(N_x, N_y);
    S2 = Matrix(N_x, N_y);
    S3 = Matrix(N_x, N_y);

    const double half_len_x = 0.5 * len_x;
    const double half_len_y = 0.5 * len_y;

    set_position_limit(-half_len_x, half_len_x, -half_len_y, half_len_y);
  }

  std::unique_ptr<Sensor> PlanarCBSSensor::clone() const
  {
    auto det = std::make_unique<PlanarCBSSensor>(len_x, len_y, dx, dy);
    det->filter_theta_enabled = filter_theta_enabled;
    det->filter_theta_min = filter_theta_min;
    det->filter_theta_max = filter_theta_max;
    det->_cache_cos_theta_min = _cache_cos_theta_min;
    det->_cache_cos_theta_max = _cache_cos_theta_max;
    det->filter_phi_enabled = filter_phi_enabled;
    det->filter_phi_min = filter_phi_min;
    det->filter_phi_max = filter_phi_max;
    det->filter_position_enabled = filter_position_enabled;
    det->filter_x_min = filter_x_min;
    det->filter_x_max = filter_x_max;
    det->filter_y_min = filter_y_min;
    det->filter_y_max = filter_y_max;
    return det;
  }

  void PlanarCBSSensor::merge_from(const Sensor &other)
  {
    const auto &o = dynamic_cast<const PlanarCBSSensor &>(other);
    hits += o.hits;
    for (int i = 0; i < N_x; ++i)
    {
      for (int j = 0; j < N_y; ++j)
      {
        S0(i, j) += o.S0(i, j);
        S1(i, j) += o.S1(i, j);
        S2(i, j) += o.S2(i, j);
        S3(i, j) += o.S3(i, j);
      }
    }
  }

  void PlanarCBSSensor::process_hit(Photon &photon, InteractionInfo &info, const Medium &medium)
  {
    const double x = info.intersection_point.x;
    const double y = info.intersection_point.y;
    const int x_idx = static_cast<int>((x + 0.5 * len_x) / dx);
    const int y_idx = static_cast<int>((y + 0.5 * len_y) / dy);
    if (x_idx < 0 || x_idx >= N_x || y_idx < 0 || y_idx >= N_y)
      return;

    const double w_sqrt = std::sqrt(photon.weight);

    // Vec3 qb = (photon.dir + photon.s_0) * photon.k;
    // Vec3 delta_r = photon.r_n - photon.r_0;
    // std::complex<double> path_phase = std::exp(std::complex<double>(0, dot(qb, delta_r)));

    // Vec3 n_0 = photon.n_0;
    // Vec3 s_0 = photon.s_0;
    // Vec3 m_0 = cross(n_0, s_0);

    // CVec2 E_fwd_local = photon.polarization;
    // CVec2 E_rev_local = photon.polarization_reverse;

    // std::complex<double> E_fwd_lab_x = (E_fwd_local.m * photon.m.x + E_fwd_local.n * photon.n.x) * info.phase * w_sqrt;
    // std::complex<double> E_fwd_lab_y = (E_fwd_local.m * photon.m.y + E_fwd_local.n * photon.n.y) * info.phase * w_sqrt;

    // std::complex<double> E_rev_lab_x = (E_rev_local.m * m_0.x + E_rev_local.n * n_0.x) * info.phase * path_phase * w_sqrt;
    // std::complex<double> E_rev_lab_y = (E_rev_local.m * m_0.y + E_rev_local.n * n_0.y) * info.phase * path_phase * w_sqrt;

    // std::complex<double> E_total_x = E_fwd_lab_x + E_rev_lab_x;
    // std::complex<double> E_total_y = E_fwd_lab_y + E_rev_lab_y;

    // const double S0_contribution = std::norm(E_total_x) + std::norm(E_total_y);
    // const double S1_contribution = std::norm(E_total_x) - std::norm(E_total_y);
    // const double S2_contribution = 2.0 * std::real(E_total_x * std::conj(E_total_y));
    // const double S3_contribution = 2.0 * std::imag(E_total_x * std::conj(E_total_y));

    // S0(x_idx, y_idx) += S0_contribution;
    // S1(x_idx, y_idx) += S1_contribution;
    // S2(x_idx, y_idx) += S2_contribution;
    // S3(x_idx, y_idx) += S3_contribution;

    hits += 1;
  }

  void PlanarCBSSensor::process_estimation(const Photon &photon, const Medium &medium)
  {
    // Not implemented for CBS sensor since it relies on interference between forward and reverse paths, which cannot be captured by estimation.
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // FarFieldFluenceSensor implementation
  // ═══════════════════════════════════════════════════════════════════════════
  FarFieldFluenceSensor::FarFieldFluenceSensor(double theta_max, double phi_max, int n_theta, int n_phi, bool estimator) : Sensor(0.0, true, estimator)
  {
    this->theta_max = theta_max;
    this->phi_max = phi_max;
    this->N_theta = n_theta;
    this->N_phi = n_phi;

    dtheta = theta_max / N_theta;
    dphi = phi_max / N_phi;

    S0 = Matrix(N_theta, N_phi);
    S1 = Matrix(N_theta, N_phi);
    S2 = Matrix(N_theta, N_phi);
    S3 = Matrix(N_theta, N_phi);

    set_theta_limit(0, theta_max);
    set_phi_limit(0, phi_max);
  }

  std::unique_ptr<Sensor> FarFieldFluenceSensor::clone() const
  {
    auto det = std::make_unique<FarFieldFluenceSensor>(theta_max, phi_max, N_theta, N_phi, estimator_enabled);
    det->filter_theta_enabled = filter_theta_enabled;
    det->filter_theta_min = filter_theta_min;
    det->filter_theta_max = filter_theta_max;
    det->_cache_cos_theta_min = _cache_cos_theta_min;
    det->_cache_cos_theta_max = _cache_cos_theta_max;
    det->filter_phi_enabled = filter_phi_enabled;
    det->filter_phi_min = filter_phi_min;
    det->filter_phi_max = filter_phi_max;
    return det;
  }

  void FarFieldFluenceSensor::merge_from(const Sensor &other)
  {
    const auto &o = dynamic_cast<const FarFieldFluenceSensor &>(other);
    hits += o.hits;
    for (int i = 0; i < N_theta; ++i)
    {
      for (int j = 0; j < N_phi; ++j)
      {
        S0(i, j) += o.S0(i, j);
        S1(i, j) += o.S1(i, j);
        S2(i, j) += o.S2(i, j);
        S3(i, j) += o.S3(i, j);
      }
    }
  }

  void FarFieldFluenceSensor::process_hit(Photon &photon, InteractionInfo &info, const Medium &medium)
  {
    Matrix P = photon.P_local;
    const Vec3 dir = {P(2, 0), P(2, 1), P(2, 2)};
    const double theta = std::acos(-dir.z);
    double phi = std::atan2(dir.y, dir.x);
    if (phi < 0)
      phi += 2.0 * M_PI;

    const int theta_idx = static_cast<int>(theta / dtheta);
    const int phi_idx = static_cast<int>(phi / dphi);
    if (theta_idx < 0 || theta_idx >= N_theta || phi_idx < 0 || phi_idx >= N_phi)
      return;

    const double w_sqrt = std::sqrt(photon.weight);
    const std::complex<double> Em_local_photon = photon.polarization.m * info.phase * w_sqrt;
    const std::complex<double> En_local_photon = photon.polarization.n * info.phase * w_sqrt;

    const std::complex<double> E_det_x = (Em_local_photon * P(0, 0) + En_local_photon * P(1, 0));
    const std::complex<double> E_det_y = (Em_local_photon * P(0, 1) + En_local_photon * P(1, 1));

    const double S0_contribution = std::norm(E_det_x) + std::norm(E_det_y);
    const double S1_contribution = std::norm(E_det_x) - std::norm(E_det_y);
    const double S2_contribution = 2.0 * std::real(E_det_x * std::conj(E_det_y));
    const double S3_contribution = 2.0 * std::imag(E_det_x * std::conj(E_det_y));

    S0(theta_idx, phi_idx) += S0_contribution;
    S1(theta_idx, phi_idx) += S1_contribution;
    S2(theta_idx, phi_idx) += S2_contribution;
    S3(theta_idx, phi_idx) += S3_contribution;

    hits += 1;
  }

  void FarFieldFluenceSensor::process_estimation(const Photon &photon, const Medium &medium)
  {
    const double EPSILON = 1e-10;
    Vec3 r_scat = photon.pos;
    double z = r_scat.z;
    double zd = origin.z;

    Matrix Pold = photon.P_local;
    Matrix Q = Matrix(3, 3);
    Matrix A = Matrix(3, 3);

    std::complex<double> E1old = photon.polarization.m;
    std::complex<double> E2old = photon.polarization.n;
    std::complex<double> Ed1;
    std::complex<double> Ed2;

    double F;

    for (int i = 0; i < N_theta; ++i)
    {
      for (int j = 0; j < N_phi; ++j)
      {
        // double theta = (i + 0.5) * dtheta;
        // double phi = (j + 0.5) * dphi;

        // Vec3 S_detector = {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), -std::cos(theta)};

        // // Distancia proyectada hacia Z=0
        // double dist_to_boundary = z - zd;
        // double attenuation = std::exp(-medium.mu_attenuation * dist_to_boundary);
        // if (attenuation < 1e-12)
        //   continue;

        // // 4. Geometría de Scattering (Igual que en el espacial, pero con s_out fijo por el loop)
        // // mu_scat = cos(angulo entre direccion actual y salida)
        // double mu_scat = dot(Vec3{photon.P_local(2, 0), photon.P_local(2, 1), photon.P_local(2, 2)}, S_detector);
        // if (mu_scat > 1.0)
        //   mu_scat = 1.0;
        // if (mu_scat < -1.0)
        //   mu_scat = -1.0;
        // double nu_scat = sqrt(1 - mu_scat * mu_scat);

        // // Vector perpendicular al plano de scattering
        // Vec3 p = cross(Vec3{photon.P_local(2, 0), photon.P_local(2, 1), photon.P_local(2, 2)}, S_detector);

        // // Proyecciones (cosphi, sinphi) del campo incidente sobre el plano
        // double sinphi = -(Pold(0, 0) * p.x + Pold(0, 1) * p.y + Pold(0, 2) * p.z); // m dot p
        // double cosphi = Pold(1, 0) * p.x + Pold(1, 1) * p.y + Pold(1, 2) * p.z;    // n dot p

        // A(0, 0) = mu_scat * cosphi;
        // A(0, 1) = mu_scat * sinphi;
        // A(0, 2) = -nu_scat;
        // A(1, 0) = -sinphi;
        // A(1, 1) = cosphi;
        // A(1, 2) = 0;
        // A(2, 0) = nu_scat * cosphi;
        // A(2, 1) = nu_scat * sinphi;
        // A(2, 2) = mu_scat;

        // matmul(A, Pold, Q);

        // CMatrix Smatrix = medium.scattering_matrix(theta, 0, photon.k);
        // double s2 = std::norm(Smatrix(0, 0));
        // double s1 = std::norm(Smatrix(1, 1));

        // double s2sq = std::norm(Smatrix(0, 0));
        // double s1sq = std::norm(Smatrix(1, 1));

        // double e1sq = std::norm(E1old);
        // double e2sq = std::norm(E2old);
        // double e12 = (E1old * conj(E2old)).real();

        // F = (s2sq * e1sq + s1sq * e2sq) * cosphi * cosphi + (s1sq * e1sq + s2sq * e2sq) * sinphi * sinphi + 2 * (s2sq - s1sq) * e12 * cosphi * sinphi;
        // Ed1 = (cosphi * E1old + sinphi * E2old) * s2 / sqrt(F);
        // Ed2 = (-sinphi * E1old + cosphi * E2old) * s1 / sqrt(F);

        // double deposit = 0.0;
        // double z = photon.pos.z;
        // double zd = origin.z;
        // double weight = photon.weight;
        // double csca = 1.0;

        // if (photon.events == 0)
        //   if (std::abs(1 - mu_scat) < 1e-11)
        //     deposit = weight * exp(-fabs((z - zd)));
        //   else
        //     deposit = 0;
        // else
        //   deposit = weight * F / csca * exp(-fabs((z - zd)));

        // double t = photon.launch_time + (photon.opticalpath / photon.velocity);
        // double td = t + fabs((z - zd));

        // std::complex<double> phase = std::exp(std::complex<double>(0, photon.k * td));
        // std::complex<double> Ex = (Ed1 * Q(0, 0) + Ed2 * Q(1, 0)) * phase;
        // std::complex<double> Ey = (Ed1 * Q(0, 1) + Ed2 * Q(1, 1)) * phase;

        // const double S0_contribution = std::norm(Ex) + std::norm(Ey);
        // const double S1_contribution = std::norm(Ex) - std::norm(Ey);
        // const double S2_contribution = 2.0 * std::real(Ex * std::conj(Ey));
        // const double S3_contribution = 2.0 * std::imag(Ex * std::conj(Ey));

        // S0(i, j) += S0_contribution * deposit;
        // S1(i, j) += S1_contribution * deposit;
        // S2(i, j) += S2_contribution * deposit;
        // S3(i, j) += S3_contribution * deposit;
      }
    }

    hits++;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // FarFieldCBSSensor implementation
  // ═══════════════════════════════════════════════════════════════════════════
  FarFieldCBSSensor::FarFieldCBSSensor(double theta_max, double phi_max, int n_theta, int n_phi, bool estimator) : Sensor(0.0, true, estimator)
  {
    this->theta_max = theta_max;
    this->phi_max = phi_max;
    this->N_theta = n_theta;
    this->N_phi = n_phi;
    dtheta = theta_max / N_theta;
    dphi = phi_max / N_phi;
    S0_coh = Matrix(N_theta, N_phi);
    S1_coh = Matrix(N_theta, N_phi);
    S2_coh = Matrix(N_theta, N_phi);
    S3_coh = Matrix(N_theta, N_phi);
    S0_incoh = Matrix(N_theta, N_phi);
    S1_incoh = Matrix(N_theta, N_phi);
    S2_incoh = Matrix(N_theta, N_phi);
    S3_incoh = Matrix(N_theta, N_phi);
  }

  void FarFieldCBSSensor::merge_from(const Sensor &other)
  {
    const auto &o = dynamic_cast<const FarFieldCBSSensor &>(other);
    hits += o.hits;
    for (int i = 0; i < N_theta; ++i)
    {
      for (int j = 0; j < N_phi; ++j)
      {
        S0_coh(i, j) += o.S0_coh(i, j);
        S1_coh(i, j) += o.S1_coh(i, j);
        S2_coh(i, j) += o.S2_coh(i, j);
        S3_coh(i, j) += o.S3_coh(i, j);
        S0_incoh(i, j) += o.S0_incoh(i, j);
        S1_incoh(i, j) += o.S1_incoh(i, j);
        S2_incoh(i, j) += o.S2_incoh(i, j);
        S3_incoh(i, j) += o.S3_incoh(i, j);
      }
    }
  }

  std::unique_ptr<Sensor> FarFieldCBSSensor::clone() const
  {
    auto det = std::make_unique<FarFieldCBSSensor>(theta_max, phi_max, N_theta, N_phi, estimator_enabled);
    det->filter_theta_enabled = filter_theta_enabled;
    det->filter_theta_min = filter_theta_min;
    det->filter_theta_max = filter_theta_max;
    det->_cache_cos_theta_min = _cache_cos_theta_min;
    det->_cache_cos_theta_max = _cache_cos_theta_max;
    det->filter_phi_enabled = filter_phi_enabled;
    det->filter_phi_min = filter_phi_min;
    det->filter_phi_max = filter_phi_max;
    det->filter_position_enabled = filter_position_enabled;
    det->filter_x_min = filter_x_min;
    det->filter_x_max = filter_x_max;
    det->filter_y_min = filter_y_min;
    det->filter_y_max = filter_y_max;
    det->theta_pp_max = theta_pp_max;
    det->theta_stride = theta_stride;
    det->phi_stride = phi_stride;
    return det;
  }

  void FarFieldCBSSensor::process_hit(Photon &photon, InteractionInfo &info, const Medium &medium)
  {
    // CBS requires at least 2 scattering events to form a time-reversed path pair.
    if (photon.events < 2)
      return;

    // Compute the reverse-path electric field via the 3-stage algorithm.
    // After this call, photon.polarization_reverse holds the reverse Jones vector.
    coherent_calculation(photon, medium);

    // --- Extract exit direction and far-field angular coordinates ---
    // Local frame at the detection point: rows of P are (m, n, s).
    const Matrix &P = photon.P_local;
    const Vec3 s_out{P(2, 0), P(2, 1), P(2, 2)}; // Exit direction

    // Far-field angles: theta=0 corresponds to exact backscattering (-z direction).
    const double theta = std::acos(-s_out.z);
    double phi = std::atan2(s_out.y, s_out.x);
    if (phi < 0)
      phi += 2.0 * M_PI;

    const int theta_idx = static_cast<int>(theta / dtheta);
    const int phi_idx = static_cast<int>(phi / dphi);
    if (theta_idx < 0 || theta_idx >= N_theta || phi_idx < 0 || phi_idx >= N_phi)
      return;

    // --- CBS geometric phase factor ---
    // The phase difference between forward and reverse paths arises from
    // the spatial separation of the first (r_0) and last (r_n) scattering events:
    //   phase = exp(i * k * (s_out + s_in) . (r_n - r_0))
    // At exact backscattering (s_out = -s_in), this phase vanishes and
    // the two paths interfere constructively, producing the CBS cone.
    const Vec3 s_in{photon.P0(2, 0), photon.P0(2, 1), photon.P0(2, 2)};
    const Vec3 qb = (s_out + s_in) * photon.k;
    const Vec3 delta_r = photon.r_n - photon.r_0;
    const std::complex<double> path_phase = std::exp(std::complex<double>(0, dot(qb, delta_r)));

    // --- Common amplitude factor ---
    // Weight-corrected amplitude with propagation phase at the detector plane.
    const double w_sqrt = std::sqrt(photon.weight);
    const std::complex<double> amp = info.phase * w_sqrt;

    // --- Forward field (already in the local frame at exit) ---
    // The forward polarization is tracked continuously during transport.
    const std::complex<double> Em_f = photon.polarization.m * amp;
    const std::complex<double> En_f = photon.polarization.n * amp;

    // Project local Jones vector (m,n) components to lab frame (x,y).
    const std::complex<double> Efx = Em_f * P(0, 0) + En_f * P(1, 0);
    const std::complex<double> Efy = Em_f * P(0, 1) + En_f * P(1, 1);

    // --- Reverse field (computed by coherent_calculation) ---
    // The reverse field includes the CBS geometric phase from the
    // spatial separation of the first and last scattering sites.
    const std::complex<double> Em_r = photon.polarization_reverse.m * amp * path_phase;
    const std::complex<double> En_r = photon.polarization_reverse.n * amp * path_phase;

    const std::complex<double> Erx = Em_r * P(0, 0) + En_r * P(1, 0);
    const std::complex<double> Ery = Em_r * P(0, 1) + En_r * P(1, 1);

    // --- Coherent Stokes: |E_forward + E_reverse|^2 ---
    // This includes the interference term that produces the CBS enhancement.
    const std::complex<double> Etx = Efx + Erx;
    const std::complex<double> Ety = Efy + Ery;

    const double S0c = std::norm(Etx) + std::norm(Ety);
    const double S1c = std::norm(Etx) - std::norm(Ety);
    const double S2c = 2.0 * std::real(Etx * std::conj(Ety));
    const double S3c = 2.0 * std::imag(Etx * std::conj(Ety));

    S0_coh(theta_idx, phi_idx) += S0c;
    S1_coh(theta_idx, phi_idx) += S1c;
    S2_coh(theta_idx, phi_idx) += S2c;
    S3_coh(theta_idx, phi_idx) += S3c;

    // --- Incoherent Stokes: |E_forward|^2 + |E_reverse|^2 ---
    // No interference; serves as the baseline background intensity.
    const double S0i = (std::norm(Efx) + std::norm(Efy)) + (std::norm(Erx) + std::norm(Ery));
    const double S1i = (std::norm(Efx) - std::norm(Efy)) + (std::norm(Erx) - std::norm(Ery));
    const double S2i = 2.0 * (std::real(Efx * std::conj(Efy)) + std::real(Erx * std::conj(Ery)));
    const double S3i = 2.0 * (std::imag(Efx * std::conj(Efy)) + std::imag(Erx * std::conj(Ery)));

    S0_incoh(theta_idx, phi_idx) += S0i;
    S1_incoh(theta_idx, phi_idx) += S1i;
    S2_incoh(theta_idx, phi_idx) += S2i;
    S3_incoh(theta_idx, phi_idx) += S3i;

    hits += 1;
  }

  void FarFieldCBSSensor::process_estimation(const Photon &photon, const Medium &medium)
  {
    // Para CBS por estimación: necesitas al menos 1 scatter previo,
    // porque el scatter estimado sería el segundo total.
    if (photon.events < 1)
      return;

    // Cache I_norm para este k
    if (_I_norm < 0.0 || std::abs(_I_norm_k - photon.k) > 1e-12)
    {
      _I_norm = compute_I_norm(medium, photon.k);
      _I_norm_k = photon.k;
      if (_I_norm < 1e-300)
        return;
    }

    // Base angular alrededor del eje de backscattering (-s_in)
    const Vec3 e1 = row_vec3(photon.P0, 0);
    const Vec3 e2 = row_vec3(photon.P0, 1);
    const Vec3 s_in = row_vec3(photon.P0, 2);
    const Vec3 e3 = s_in * (-1.0); // theta_det=0 es backscatter exacto

    // Frame actual antes del scatter estimado
    const Matrix &Pcur = photon.P_local;
    const Vec3 m_cur = row_vec3(Pcur, 0);
    const Vec3 n_cur = row_vec3(Pcur, 1);
    const Vec3 s_cur = row_vec3(Pcur, 2);

    // Tmid para n = events+1  => interior = J_events ... J2
    CMatrix Tmid = CMatrix::identity(2);
    if (photon.events >= 2 && photon.has_T_prev)
    {
      matcmul(photon.matrix_T_buffer, photon.matrix_T, Tmid);
    }

    // Control de región y subsampling
    const double th_max = (theta_pp_max > 0.0) ? std::min(theta_pp_max, theta_max) : theta_max;
    const int i_max = std::min(N_theta, static_cast<int>(th_max / dtheta));

    // Pre-factor de “sobrevive al scatter” (mu_s/mu_t)
    const double w_scatter = photon.weight * (medium.mu_scattering / medium.mu_attenuation);
    if (w_scatter < 1e-300)
      return;

    // Plano detector (usa tu Sensor::origin y normal)
    const Vec3 p0 = origin;
    const Vec3 nd = normal;

    for (int i = 0; i < i_max; i += std::max(1, theta_stride))
    {
      // sólido del bin (exacto por bordes)
      const double th0 = i * dtheta;
      const double th1 = (i + 1) * dtheta;
      const double dOmega_theta = (std::cos(th0) - std::cos(th1)); // falta *dphi

      const double th_det = (i + 0.5) * dtheta;
      const double s_th = std::sin(th_det);
      const double c_th = std::cos(th_det);

      for (int j = 0; j < N_phi; j += std::max(1, phi_stride))
      {
        const double ph_det = (j + 0.5) * dphi;
        const double c_ph = std::cos(ph_det);
        const double s_ph = std::sin(ph_det);

        // Dirección del bin en lab coords:
        // s_out = sinθ cosφ e1 + sinθ sinφ e2 + cosθ e3
        Vec3 s_out = e1 * (s_th * c_ph) + e2 * (s_th * s_ph) + e3 * (c_th);

        // 1) Intersección con el plano detector
        double L;
        if (!intersect_plane(photon.pos, s_out, p0, nd, L))
          continue;

        // 2) Transmitancia sin scatter adicional: exp(-mu_t L)
        const double Tr = std::exp(-medium.mu_attenuation * L);
        if (Tr < 1e-20)
          continue;

        // 3) Geometría del scatter estimado: (s_cur -> s_out)
        const double mu = clamp_pm1(dot(s_cur, s_out));
        const double th_scat = std::acos(mu);

        // vector normal al plano de scattering
        Vec3 p = cross(s_cur, s_out);
        const double pn = norm(p);
        if (pn < 1e-14)
          continue; // plano degenerado (muy raro, pero pasa en forward/backward exacto)
        Vec3 p_hat = p * (1.0 / pn);

        // Define phi_scat relativo a la base actual (m_cur,n_cur)
        const double sin_phi = -dot(m_cur, p_hat);
        const double cos_phi = dot(n_cur, p_hat);

        // 4) Matriz S(th_scat)
        const CMatrix S = medium.scattering_matrix(th_scat, 0.0, photon.k);
        const double s22 = std::norm(S(0, 0));
        const double s11 = std::norm(S(1, 1));

        // 5) F(phi) (tu misma expresión) -> densidad angular correcta
        const std::complex<double> Em = photon.polarization.m;
        const std::complex<double> En = photon.polarization.n;

        const double Emm = std::norm(Em);
        const double Enn = std::norm(En);

        const double c2 = cos_phi * cos_phi;
        const double s2 = sin_phi * sin_phi;

        const double F =
            Emm * (s22 * c2 + s11 * s2) +
            Enn * (s22 * s2 + s11 * c2) +
            2.0 * std::real(Em * std::conj(En)) * (s22 - s11) * sin_phi * cos_phi;

        if (F < 1e-300)
          continue;

        // p(Ω) = F / (π I_norm)   (densidad por sólido)
        const double pOmega = F / (M_PI * _I_norm);

        // ΔΩ_bin ~ ΔΩ_theta * dphi  (exacto en θ, uniforme en φ)
        const double dOmega = dOmega_theta * dphi;
        const double prob_bin = pOmega * dOmega;
        if (prob_bin <= 0.0)
          continue;

        // 6) Peso esperado en ese bin
        const double w_bin = w_scatter * Tr * prob_bin;
        if (w_bin < 1e-300)
          continue;

        const std::complex<double> amp = std::sqrt(w_bin) * std::exp(std::complex<double>(0.0, photon.k * L));

        // 7) Frame de salida P_exit consistente con tu convención (A_update)
        const double cos_ths = std::cos(th_scat);
        const double sin_ths = std::sin(th_scat);

        Matrix A(3, 3);
        A(0, 0) = cos_ths * cos_phi;
        A(0, 1) = cos_ths * sin_phi;
        A(0, 2) = -sin_ths;
        A(1, 0) = -sin_phi;
        A(1, 1) = cos_phi;
        A(1, 2) = 0;
        A(2, 0) = sin_ths * cos_phi;
        A(2, 1) = sin_ths * sin_phi;
        A(2, 2) = cos_ths;

        Matrix P_exit(3, 3);
        matmul(A, Pcur, P_exit);

        // 8) Forward field local (en base P_exit) usando operador normalizado
        const CVec2 Ef_loc = apply_scatter_normalized(S, cos_phi, sin_phi, photon.polarization);

        // 9) Reverse local via reciprocity (en base P_exit)
        const CVec2 Er_loc = coherent_estimation_partial(photon, medium, Pcur, P_exit, Tmid);

        // 10) Fase CBS geométrica: exp(i k (s_out + s_in)·(r_n - r_0))
        const Vec3 qb = (s_out + s_in) * photon.k;
        const Vec3 delta_r = photon.pos - photon.r_0; // r_n estimado = pos actual
        const std::complex<double> path_phase = std::exp(std::complex<double>(0.0, dot(qb, delta_r)));

        // 11) Proyección a lab (x,y) con P_exit
        const std::complex<double> Efx = amp * (Ef_loc.m * P_exit(0, 0) + Ef_loc.n * P_exit(1, 0));
        const std::complex<double> Efy = amp * (Ef_loc.m * P_exit(0, 1) + Ef_loc.n * P_exit(1, 1));

        const std::complex<double> Erx = amp * path_phase * (Er_loc.m * P_exit(0, 0) + Er_loc.n * P_exit(1, 0));
        const std::complex<double> Ery = amp * path_phase * (Er_loc.m * P_exit(0, 1) + Er_loc.n * P_exit(1, 1));

        // Coherente
        const std::complex<double> Etx = Efx + Erx;
        const std::complex<double> Ety = Efy + Ery;

        const double S0c = std::norm(Etx) + std::norm(Ety);
        const double S1c = std::norm(Etx) - std::norm(Ety);
        const double S2c = 2.0 * std::real(Etx * std::conj(Ety));
        const double S3c = 2.0 * std::imag(Etx * std::conj(Ety));

        S0_coh(i, j) += S0c;
        S1_coh(i, j) += S1c;
        S2_coh(i, j) += S2c;
        S3_coh(i, j) += S3c;

        // Incoherente (baseline)
        const double S0i = (std::norm(Efx) + std::norm(Efy)) + (std::norm(Erx) + std::norm(Ery));
        const double S1i = (std::norm(Efx) - std::norm(Efy)) + (std::norm(Erx) - std::norm(Ery));
        const double S2i = 2.0 * (std::real(Efx * std::conj(Efy)) + std::real(Erx * std::conj(Ery)));
        const double S3i = 2.0 * (std::imag(Efx * std::conj(Efy)) + std::imag(Erx * std::conj(Ery)));

        S0_incoh(i, j) += S0i;
        S1_incoh(i, j) += S1i;
        S2_incoh(i, j) += S2i;
        S3_incoh(i, j) += S3i;
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // CBS free functions: coherent_estimation & coherent_calculation
  //
  // These implement the three-stage reverse-path algorithm for coherent
  // backscattering (CBS). The reverse path visits the same scattering
  // sites as the forward path but in reversed order. By the reciprocity
  // theorem, the interior segment of the reverse path can be obtained
  // from the forward transfer matrix T via Q * T^T * Q, avoiding the
  // need to re-trace the full path.
  //
  // The three stages are:
  //   A) First reverse scatter at r_n: (s_in=s0) -> (s_out=-s_{n-1})
  //   B) Middle segment: apply Q * T^T * Q to propagate through
  //      the bulk in reverse via the reciprocity shortcut.
  //   C) Last reverse scatter at r_1: (s_in=-s1) -> (s_out=s_n)
  //
  // coherent_estimation: variant for next-event estimation (forced
  //   detection), where the last scattering frame is given externally.
  // coherent_calculation: variant for direct detection, where the
  //   last scattering frame is photon.Pn (the actual exit frame).
  // ═══════════════════════════════════════════════════════════════════════════

  CVec2 coherent_estimation_partial(
      const Photon &photon,
      const Medium &medium,
      const Matrix &P_last_in,  // frame de s_{n-1} (antes del scatter estimado)
      const Matrix &P_last_out, // frame de s_n     (después del scatter estimado)
      const CMatrix &Tmid)      // J_{n-1}...J_2  (para n=events+1)
  {
    // Direcciones clave
    const Vec3 s0 = row_vec3(photon.P0, 2);
    const Vec3 s1 = row_vec3(photon.P1, 2);
    const Vec3 snm1 = row_vec3(P_last_in, 2);
    const Vec3 sn = row_vec3(P_last_out, 2);

    // Bases transversales
    const Vec3 m0 = row_vec3(photon.P0, 0);
    const Vec3 n0 = row_vec3(photon.P0, 1);

    const Vec3 m1 = row_vec3(photon.P1, 0);
    const Vec3 n1 = row_vec3(photon.P1, 1);

    const Vec3 mnm1 = row_vec3(P_last_in, 0);
    const Vec3 nnm1 = row_vec3(P_last_in, 1);

    const Vec3 mn = row_vec3(P_last_out, 0);
    const Vec3 nn = row_vec3(P_last_out, 1);

    // Q = diag(1,-1)
    CMatrix Q(2, 2);
    Q(0, 0) = 1;
    Q(0, 1) = 0;
    Q(1, 0) = 0;
    Q(1, 1) = -1;

    // T^T (NO conjugada)
    CMatrix Tt(2, 2);
    Tt(0, 0) = Tmid(0, 0);
    Tt(0, 1) = Tmid(1, 0);
    Tt(1, 0) = Tmid(0, 1);
    Tt(1, 1) = Tmid(1, 1);

    // ===========================
    // Stage A: scatter en r_n (estimado)
    // s_in = s0, s_out = -s_{n-1}
    // ===========================
    const Vec3 s_in_a = s0;
    const Vec3 s_out_a = snm1 * (-1.0);

    const double th_a = std::acos(clamp_pm1(dot(s_in_a, s_out_a)));
    const CMatrix S_a = medium.scattering_matrix(th_a, 0.0, photon.k);

    const Vec3 nprime = safe_unit(cross(s_in_a, s_out_a), n0);
    const Vec3 mprime_in = safe_unit(cross(nprime, s_in_a), m0);
    const Vec3 mprime_out = safe_unit(cross(nprime, s_out_a), mnm1);

    // Rotación desde (m0,n0) hacia el plano
    const CMatrix Rn = rot2(mprime_in, nprime, m0, n0);

    // extrae cos/sin de Rn (asume rotación real 2D)
    const double cphi_a = std::real(Rn(0, 0));
    const double sphi_a = std::real(Rn(0, 1));

    CVec2 E = apply_scatter_normalized(S_a, cphi_a, sphi_a, photon.initial_polarization);

    // Rotación hacia base del mid-segment: (m_{n-1}, -n_{n-1})
    const Vec3 n_to_mid = nnm1 * (-1.0);
    const CMatrix Rnp = rot2(mnm1, n_to_mid, mprime_out, nprime);
    E = apply2(Rnp, E);

    // ===========================
    // Stage B: Q T^T Q
    // ===========================
    E = apply2(Q, E);
    E = apply2(Tt, E);
    E = apply2(Q, E);

    // ===========================
    // Stage C: scatter en r_1
    // s_in = -s1, s_out = s_n
    // ===========================
    const Vec3 s_in_c = s1 * (-1.0);
    const Vec3 s_out_c = sn;

    const double th_c = std::acos(clamp_pm1(dot(s_in_c, s_out_c)));
    const CMatrix S_c = medium.scattering_matrix(th_c, 0.0, photon.k);

    const Vec3 npp = safe_unit(cross(s_in_c, s_out_c), n1 * (-1.0));
    const Vec3 mpp_in = safe_unit(cross(npp, s_in_c), m1);
    const Vec3 mpp_out = safe_unit(cross(npp, s_out_c), mn);

    const Vec3 n_mid_end = n1 * (-1.0);
    const CMatrix R1p = rot2(mpp_in, npp, m1, n_mid_end);

    const double cphi_c = std::real(R1p(0, 0));
    const double sphi_c = std::real(R1p(0, 1));

    E = apply_scatter_normalized(S_c, cphi_c, sphi_c, E);

    const CMatrix Rout = rot2(mn, nn, mpp_out, npp);
    E = apply2(Rout, E);

    return E; // en base (mn,nn) del frame P_last_out
  }

  void coherent_calculation(Photon &photon, const Medium &medium)
  {
    // --- Extract local frames and propagation directions ---
    // Same structure as coherent_estimation, but uses photon.Pn (actual exit
    // frame) instead of an externally-supplied estimator frame.
    Vec3 s0 = row_vec3(photon.P0, 2);    // Initial propagation direction
    Vec3 s1 = row_vec3(photon.P1, 2);    // Direction after first scatter
    Vec3 snm1 = row_vec3(photon.Pn1, 2); // Direction before last scatter
    Vec3 sn = row_vec3(photon.Pn, 2);    // Actual exit direction

    Vec3 m0 = row_vec3(photon.P0, 0);
    Vec3 n0 = row_vec3(photon.P0, 1);

    Vec3 m1 = row_vec3(photon.P1, 0);
    Vec3 n1 = row_vec3(photon.P1, 1);

    Vec3 mnm1 = row_vec3(photon.Pn1, 0);
    Vec3 nnm1 = row_vec3(photon.Pn1, 1);

    Vec3 mn = row_vec3(photon.Pn, 0);
    Vec3 nn = row_vec3(photon.Pn, 1);

    // ===========================
    // Stage A: First reverse scatter at r_n
    //   s_in  = s0       (original incidence direction)
    //   s_out = -s_{n-1} (reversed penultimate direction)
    // ===========================
    Vec3 s_in_a = s0;
    Vec3 s_out_a = snm1 * (-1.0);

    double cos_th_a = clamp_pm1(dot(s_in_a, s_out_a));
    double th_a = std::acos(cos_th_a);
    CMatrix S_a = medium.scattering_matrix(th_a, 0.0, photon.k);

    // Scattering plane normal: n' = s_in x s_out
    Vec3 nprime = safe_unit(cross(s_in_a, s_out_a), n0);
    Vec3 mprime_in = safe_unit(cross(nprime, s_in_a), m0);
    Vec3 mprime_out = safe_unit(cross(nprime, s_out_a), mnm1);

    // Rotation R(phi_n): from initial basis (m0,n0) to scattering plane
    CMatrix Rn = rot2(mprime_in, nprime, m0, n0);

    // Apply scattering matrix in the scattering plane; output in (mprime_out, nprime)
    // CVec2 E = scatter_event(S_a, Rn, photon.initial_polarization);

    // // Rotation R(phi_n'): from scattering plane to reverse mid-segment basis
    // // Target basis: (m_{n-1}, -n_{n-1}) in direction -s_{n-1}
    // Vec3 n_to_mid_start = nnm1 * (-1.0);
    // CMatrix Rnp = rot2(mnm1, n_to_mid_start, mprime_out, nprime);
    // E = apply2(Rnp, E);

    // NEW
    CMatrix raw_A(2, 2);
    matcmul(S_a, Rn, raw_A);
    CVec2 E;
    E.m = raw_A(0, 0) * photon.initial_polarization.m + raw_A(0, 1) * photon.initial_polarization.n;
    E.n = raw_A(1, 0) * photon.initial_polarization.m + raw_A(1, 1) * photon.initial_polarization.n;
    Vec3 n_to_mid_start = nnm1 * (-1.0);
    CMatrix Rnp = rot2(mnm1, n_to_mid_start, mprime_out, nprime);
    E = apply2(Rnp, E);

    // ===========================
    // Stage B: Middle segment via reciprocity shortcut
    //   E <- Q * T^T * Q * E
    // ===========================

    // --- Q matrix: parity flip for reciprocity ---
    CMatrix Q(2, 2);
    Q(0, 0) = 1;
    Q(0, 1) = 0;
    Q(1, 0) = 0;
    Q(1, 1) = -1;

    // --- T^T (transpose, NOT conjugate transpose) ---
    CMatrix Tt_raw(2, 2);
    Tt_raw(0, 0) = photon.matrix_T_raw(0, 0);
    Tt_raw(0, 1) = photon.matrix_T_raw(1, 0); // transpuesta
    Tt_raw(1, 0) = photon.matrix_T_raw(0, 1);
    Tt_raw(1, 1) = photon.matrix_T_raw(1, 1);

    E = apply2(Q, E);
    E = apply2(Tt_raw, E);
    E = apply2(Q, E);

    // E is now in basis (m1, -n1) propagating in direction -s1

    // ===========================
    // Stage C: Last reverse scatter at r_1
    //   s_in  = -s1  (reversed post-first-scatter direction)
    //   s_out = sn   (actual exit direction)
    // ===========================
    Vec3 s_in_b = s1 * (-1.0);
    Vec3 s_out_b = sn;

    double cos_th_b = clamp_pm1(dot(s_in_b, s_out_b));
    double th_b = std::acos(cos_th_b);
    CMatrix S_b = medium.scattering_matrix(th_b, 0.0, photon.k);

    // Scattering plane normal: n'' = (-s1) x sn
    Vec3 npp = safe_unit(cross(s_in_b, s_out_b), n1 * (-1.0));
    Vec3 mpp_in = safe_unit(cross(npp, s_in_b), m1);
    Vec3 mpp_out = safe_unit(cross(npp, s_out_b), mn);

    // Rotation R(phi_1'): from mid-segment basis (m1,-n1) to scattering plane
    Vec3 n_mid_end = n1 * (-1.0);
    CMatrix R1p = rot2(mpp_in, npp, m1, n_mid_end);

    // // Apply scattering; output in (mpp_out, npp)
    // E = scatter_event(S_b, R1p, E);

    // // Final rotation: from scattering plane to actual exit basis (mn, nn)
    // CMatrix Rout = rot2(mn, nn, mpp_out, npp);
    // E = apply2(Rout, E);

    // // Store the reverse-path Jones vector on the photon for use by process_hit.
    // photon.polarization_reverse = E_out;

    // NEW
    CMatrix raw_C(2, 2);
    matcmul(S_b, R1p, raw_C);
    CVec2 E_out;
    E_out.m = raw_C(0, 0) * E.m + raw_C(0, 1) * E.n;
    E_out.n = raw_C(1, 0) * E.m + raw_C(1, 1) * E.n;
    CMatrix Rout = rot2(mn, nn, mpp_out, npp);
    E_out = apply2(Rout, E_out);
    double norm_r = std::sqrt(std::norm(E_out.m) + std::norm(E_out.n));
    if (norm_r > 1e-300)
    {
      photon.polarization_reverse.m = E_out.m / norm_r;
      photon.polarization_reverse.n = E_out.n / norm_r;
    }
    else
    {
      photon.polarization_reverse.m = 0;
      photon.polarization_reverse.n = 0;
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // StatisticsSensor implementation
  // ═══════════════════════════════════════════════════════════════════════════
  StatisticsSensor::StatisticsSensor(double z, bool absorb) : Sensor(z, absorb, false)
  {
    events_histogram = std::vector<int>();
    theta_histogram = std::vector<int>();
    phi_histogram = std::vector<int>();
    depth_histogram = std::vector<int>();
    time_histogram = std::vector<int>();
    weight_histogram = std::vector<int>();
  }

  std::unique_ptr<Sensor> StatisticsSensor::clone() const
  {
    auto det = std::make_unique<StatisticsSensor>(origin.z, absorb_photons);
    det->filter_theta_enabled = filter_theta_enabled;
    det->filter_theta_min = filter_theta_min;
    det->filter_theta_max = filter_theta_max;
    det->_cache_cos_theta_min = _cache_cos_theta_min;
    det->_cache_cos_theta_max = _cache_cos_theta_max;
    det->filter_phi_enabled = filter_phi_enabled;
    det->filter_phi_min = filter_phi_min;
    det->filter_phi_max = filter_phi_max;
    det->filter_position_enabled = filter_position_enabled;
    det->filter_x_min = filter_x_min;
    det->filter_x_max = filter_x_max;
    det->filter_y_min = filter_y_min;
    det->filter_y_max = filter_y_max;

    det->events_histogram_bins_set = events_histogram_bins_set;
    det->max_events = max_events;
    det->theta_histogram_bins_set = theta_histogram_bins_set;
    det->min_theta = min_theta;
    det->max_theta = max_theta;
    det->n_bins_theta = n_bins_theta;
    det->dtheta = dtheta;
    det->phi_histogram_bins_set = phi_histogram_bins_set;
    det->min_phi = min_phi;
    det->max_phi = max_phi;
    det->n_bins_phi = n_bins_phi;
    det->dphi = dphi;
    det->depth_histogram_bins_set = depth_histogram_bins_set;
    det->max_depth = max_depth;
    det->n_bins_depth = n_bins_depth;
    det->ddepth = ddepth;
    det->time_histogram_bins_set = time_histogram_bins_set;
    det->max_time = max_time;
    det->n_bins_time = n_bins_time;
    det->dtime = dtime;
    det->weight_histogram_bins_set = weight_histogram_bins_set;
    det->max_weight = max_weight;
    det->n_bins_weight = n_bins_weight;
    det->dweight = dweight;

    if (events_histogram_bins_set)
    {
      det->events_histogram.resize(max_events, 0);
    }
    if (theta_histogram_bins_set)
    {
      det->theta_histogram.resize(n_bins_theta, 0);
    }
    if (phi_histogram_bins_set)
    {
      det->phi_histogram.resize(n_bins_phi, 0);
    }
    if (depth_histogram_bins_set)
    {
      det->depth_histogram.resize(n_bins_depth, 0);
    }
    if (time_histogram_bins_set)
    {
      det->time_histogram.resize(n_bins_time, 0);
    }
    if (weight_histogram_bins_set)
    {
      det->weight_histogram.resize(n_bins_weight, 0);
    }
    return det;
  }

  void StatisticsSensor::merge_from(const Sensor &other)
  {
    const auto &o = dynamic_cast<const StatisticsSensor &>(other);
    hits += o.hits;

    if (events_histogram_bins_set && o.events_histogram_bins_set)
    {
      for (size_t i = 0; i < events_histogram.size(); ++i)
      {
        events_histogram[i] += o.events_histogram[i];
      }
    }
    if (theta_histogram_bins_set && o.theta_histogram_bins_set)
    {
      for (size_t i = 0; i < theta_histogram.size(); ++i)
      {
        theta_histogram[i] += o.theta_histogram[i];
      }
    }
    if (phi_histogram_bins_set && o.phi_histogram_bins_set)
    {
      for (size_t i = 0; i < phi_histogram.size(); ++i)
      {
        phi_histogram[i] += o.phi_histogram[i];
      }
    }
    if (depth_histogram_bins_set && o.depth_histogram_bins_set)
    {
      for (size_t i = 0; i < depth_histogram.size(); ++i)
      {
        depth_histogram[i] += o.depth_histogram[i];
      }
    }
    if (time_histogram_bins_set && o.time_histogram_bins_set)
    {
      for (size_t i = 0; i < time_histogram.size(); ++i)
      {
        time_histogram[i] += o.time_histogram[i];
      }
    }
    if (weight_histogram_bins_set && o.weight_histogram_bins_set)
    {
      for (size_t i = 0; i < weight_histogram.size(); ++i)
      {
        weight_histogram[i] += o.weight_histogram[i];
      }
    }
  }

  void StatisticsSensor::process_hit(Photon &photon, InteractionInfo &info, const Medium &medium)
  {
    if (events_histogram_bins_set)
    {
      int events = photon.events;
      if (events >= 0 && events < max_events)
      {
        events_histogram[events]++;
      }
    }
    if (theta_histogram_bins_set)
    {
      double theta = std::acos(-photon.P_local(2, 2));
      if (theta >= min_theta && theta < max_theta)
      {
        int idx = static_cast<int>((theta - min_theta) / dtheta);
        theta_histogram[idx]++;
      }
    }
    if (phi_histogram_bins_set)
    {
      double phi = std::atan2(photon.P_local(2, 1), photon.P_local(2, 0));
      if (phi < 0)
        phi += 2.0 * M_PI;
      if (phi >= min_phi && phi < max_phi)
      {
        int idx = static_cast<int>((phi - min_phi) / dphi);
        phi_histogram[idx]++;
      }
    }
    if (depth_histogram_bins_set)
    {
      double depth = photon.penetration_depth;
      if (depth >= 0 && depth < max_depth)
      {
        int idx = static_cast<int>(depth / ddepth);
        depth_histogram[idx]++;
      }
    }
    if (time_histogram_bins_set)
    {
      double time = photon.launch_time + (photon.opticalpath / photon.velocity);
      if (time >= 0 && time < max_time)
      {
        int idx = static_cast<int>(time / dtime);
        time_histogram[idx]++;
      }
    }
    if (weight_histogram_bins_set)
    {
      double weight = photon.weight;
      if (weight >= 0 && weight < max_weight)
      {
        int idx = static_cast<int>(weight / dweight);
        weight_histogram[idx]++;
      }
    }

    hits += 1;
  }

  void StatisticsSensor::set_events_histogram_bins(int max_events)
  {
    this->max_events = max_events;
    events_histogram_bins_set = true;
    events_histogram.resize(max_events, 0);
  }

  void StatisticsSensor::set_theta_histogram_bins(double min_theta, double max_theta, int n_bins)
  {
    this->min_theta = min_theta;
    this->max_theta = max_theta;
    this->n_bins_theta = n_bins;
    dtheta = (max_theta - min_theta) / n_bins;
    theta_histogram_bins_set = true;
    theta_histogram.resize(n_bins, 0);
  }

  void StatisticsSensor::set_phi_histogram_bins(double min_phi, double max_phi, int n_bins)
  {
    this->min_phi = min_phi;
    this->max_phi = max_phi;
    this->n_bins_phi = n_bins;
    dphi = (max_phi - min_phi) / n_bins;
    phi_histogram_bins_set = true;
    phi_histogram.resize(n_bins, 0);
  }

  void StatisticsSensor::set_depth_histogram_bins(double max_depth, int n_bins)
  {
    this->max_depth = max_depth;
    this->n_bins_depth = n_bins;
    ddepth = max_depth / n_bins;
    depth_histogram_bins_set = true;
    depth_histogram.resize(n_bins, 0);
  }

  void StatisticsSensor::set_time_histogram_bins(double max_time, int n_bins)
  {
    this->max_time = max_time;
    this->n_bins_time = n_bins;
    dtime = max_time / n_bins;
    time_histogram_bins_set = true;
    time_histogram.resize(n_bins, 0);
  }

  void StatisticsSensor::set_weight_histogram_bins(double max_weight, int n_bins)
  {
    this->max_weight = max_weight;
    this->n_bins_weight = n_bins;
    dweight = max_weight / n_bins;
    weight_histogram_bins_set = true;
    weight_histogram.resize(n_bins, 0);
  }

} // namespace luminis::core
