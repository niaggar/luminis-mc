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
 * - Three-stage reverse path computation for CBS (reverse_field)
 * - CBS geometric phase and coherent/incoherent Stokes decomposition
 *
 * @see detector.hpp for class declarations and detailed API documentation.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <luminis/core/detector.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/core/medium.hpp>
#include <luminis/core/sample.hpp>
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

  // Optical depth along a straight ray from `start` toward the detector plane,
  // total distance L_total, summed layer by layer:  tau = Σ_i μ_{t,i} · L_i.
  // In a single-layer sample this reduces to μ_t · L_total; in a stratified
  // sample it correctly accounts for each crossed layer's extinction.
  static double optical_depth_to_detector(const Sample &medium, const Vec3 &start,
                                          const Vec3 &dir, double L_total)
  {
    double tau = 0.0, traveled = 0.0;
    Vec3 p = start;
    std::size_t layer = medium.get_layer_index_at(p.z);
    if (layer >= medium.size())
      return 0.0; // start already outside the stack

    // Fast path: a single-layer sample has no internal interfaces to cross, so
    // the optical depth is simply μ_t · L over the whole ray. This is the common
    // case and is hit once per angular bin by the CBS estimator.
    if (medium.size() == 1)
      return medium.get_layer(layer).mu_attenuation() * L_total;
    const double dz = dir.z;

    while (traveled < L_total - 1e-15)
    {
      const double remaining = L_total - traveled;
      const double mu = medium.get_layer(layer).mu_attenuation();

      double seg = remaining;
      if (std::abs(dz) > 1e-15)
      {
        auto iface = medium.find_next_interface(p.z, p.z + dz * remaining);
        if (iface.has_value())
        {
          const double s = (iface.value() - p.z) / dz; // distance to the interface
          if (s > 1e-15 && s < remaining)
            seg = s;
        }
      }

      tau += mu * seg;
      p.x += dir.x * seg;
      p.y += dir.y * seg;
      p.z += dir.z * seg;
      traveled += seg;

      if (seg >= remaining - 1e-15)
        break; // reached the detector plane
      const double nudge = (dz > 0) ? 1e-12 : -1e-12;
      std::size_t next = medium.get_layer_index_at(p.z + nudge);
      if (next >= medium.size())
        break; // left the stack
      layer = next;
    }
    return tau;
  }

  // Implicit phase-function normalization carried by S_matrix:
  //   I_norm = ∫_0^π (|S11|^2 + |S22|^2) sinθ dθ
  static double compute_I_norm(const ScatteringMedium &medium, double k, int n = 2048)
  {
    double acc = 0.0;
    const double dth = M_PI / n;
    for (int i = 0; i < n; i++)
    {
      const double th = (i + 0.5) * dth;
      const CMatrix S = medium.scattering_matrix(th, 0.0);
      const double s22 = std::norm(S(0, 0));
      const double s11 = std::norm(S(1, 1));
      acc += (s11 + s22) * std::sin(th);
    }
    return acc * dth;
  }

  // Apply T = S*R and renormalize by F (same formula as the transport kernel).
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

  bool SensorsGroup::record_hit(Photon &photon, const Sample &medium)
  {
    bool photon_killed = false;

    // Determine the z-range traversed by the photon in this step.
    // Skip if the photon moved purely within a horizontal plane (no z displacement).
    double z1 = photon.prev_pos.z;
    double z2 = photon.pos.z;
    if (std::abs(z2 - z1) < 1e-12)
      return false;

    // Determine the crossing direction from the photon's displacement.
    const CrossingDirection crossing_dir = (z2 > z1) ? CrossingDirection::Forward : CrossingDirection::Backward;

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
        // The photon may have overshot the plane, so we subtract the extra
        // distance to get the path length at the intersection. The phase uses the
        // OPTICAL path (Σ nᵢ·dᵢ); the overshoot is removed with the local n.
        const Vec3 hit_point = xn + d * t;
        const double correction_distance = luminis::math::norm(hit_point - xf);
        const double n_local = 1.0 / medium.light_speed_in_medium();
        double optical_path_correction = photon.optical_path;
        double opticalpath_at_hit = photon.opticalpath;
        if (correction_distance > 0)
        {
          // Remove the overshoot beyond the plane from BOTH the optical path
          // (phase, weighted by the local n) and the geometric path (time-of-flight).
          optical_path_correction -= correction_distance * n_local;
          opticalpath_at_hit -= correction_distance;
        }

        // Build the interaction info with the corrected propagation phase.
        InteractionInfo info;
        info.intersection_point = hit_point;
        info.phase = std::exp(std::complex<double>(0, photon.k * optical_path_correction));
        info.crossing_direction = crossing_dir;
        info.opticalpath_at_hit = opticalpath_at_hit;

        for (Sensor *det : it->second)
        {
          // Estimator-mode sensors are detected exclusively via run_estimators()
          // (process_estimation at every scattering vertex). Also recording the
          // physical crossing here would double-count and, worse, inject the
          // high-variance analog contribution — a single photon dumping its full
          // weight into one bin — as spikes on top of the smooth estimator curve.
          // The photon still terminates normally via the boundary (is_inside) check.
          if (det->estimator_enabled)
            continue;

          Vec3 direction = {photon.P_local(2, 0), photon.P_local(2, 1), photon.P_local(2, 2)};
          const bool valid_photon = det->check_conditions(hit_point, direction, crossing_dir, photon.events);
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

  void SensorsGroup::run_estimators(const Photon &photon, const Sample &medium)
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

  void Sensor::set_direction_limit(CrossingDirection dir)
  {
    filter_direction_enabled = true;
    filter_direction = dir;
  }

  void Sensor::set_events_limit(int min, int max)
  {
    filter_events_enabled = true;
    filter_events_min = min;
    filter_events_max = max;
  }

  bool Sensor::check_conditions(const Vec3 &hit_point, const Vec3 &hit_direction, CrossingDirection crossing_dir, int events) const
  {
    // Apply all enabled filters in sequence (direction, theta, phi, position).
    // Returns false as soon as any filter rejects the photon.
    if (filter_direction_enabled)
    {
      if (filter_direction != CrossingDirection::Both && filter_direction != crossing_dir)
        return false;
    }

    if (filter_events_enabled)
    {
      if (events < filter_events_min || events > filter_events_max)
        return false;
    }

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

  void Sensor::process_estimation(const Photon &photon, const Sample &medium)
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
    det->filter_direction_enabled = filter_direction_enabled;
    det->filter_direction = filter_direction;
    det->filter_events_enabled = filter_events_enabled;
    det->filter_events_min = filter_events_min;
    det->filter_events_max = filter_events_max;
    return det;
  }

  void PhotonRecordSensor::merge_from(const Sensor &other)
  {
    const auto &o = dynamic_cast<const PhotonRecordSensor &>(other);
    hits += o.hits;
    recorded_photons.reserve(recorded_photons.size() + o.recorded_photons.size());
    recorded_photons.insert(recorded_photons.end(), o.recorded_photons.begin(), o.recorded_photons.end());
  }

  void PhotonRecordSensor::process_hit(Photon &photon, InteractionInfo &info, const Sample &medium)
  {
    hits += 1;

    PhotonRecord photon_rec{};
    photon_rec.events = photon.events;
    photon_rec.penetration_depth = photon.penetration_depth;
    photon_rec.launch_time = photon.launch_time;
    photon_rec.arrival_time = photon.launch_time + (info.opticalpath_at_hit / photon.velocity);
    photon_rec.opticalpath = info.opticalpath_at_hit;
    photon_rec.weight = photon.weight;
    photon_rec.k = photon.k;
    photon_rec.position_detector = info.intersection_point;
    photon_rec.position_first_scattering = photon.r_1;
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
    det->filter_direction_enabled = filter_direction_enabled;
    det->filter_direction = filter_direction;
    det->filter_events_enabled = filter_events_enabled;
    det->filter_events_min = filter_events_min;
    det->filter_events_max = filter_events_max;
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

  void PlanarFieldSensor::process_hit(Photon &photon, InteractionInfo &info, const Sample &medium)
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

  // TODO: Implement
  void PlanarFieldSensor::process_estimation(const Photon &photon, const Sample &medium)
  {
    // const double x = photon.pos.x;
    // const double y = photon.pos.y;
    // const int x_idx = static_cast<int>((x + 0.5 * len_x) / dx);
    // const int y_idx = static_cast<int>((y + 0.5 * len_y) / dy);
    // if (x_idx < 0 || x_idx >= N_x || y_idx < 0 || y_idx >= N_y)
    //   return;

    // // Local base of the detector plane
    // double ud = 0;
    // double vd = 0;
    // double wd = 1;

    // Matrix Pold = photon.P_local;
    // Matrix Q = Matrix(3, 3);
    // Matrix A = Matrix(3, 3);

    // double mu = Pold(2, 0) * ud + Pold(2, 1) * vd + Pold(2, 2) * wd; // Cosine of angle between photon direction and detector normal
    // double nu = sqrt(1 - mu * mu);                                   // Sine of the same angle
    // double F = 1;

    // std::complex<double> E1old = photon.polarization.m;
    // std::complex<double> E2old = photon.polarization.n;
    // std::complex<double> Ed1;
    // std::complex<double> Ed2;
    // Vec3 p;

    // const ScatteringMedium *current_medium = medium.get_layer(photon.current_layer).medium;

    // if (std::abs(1 - mu) < 1e-11)
    // {
    //   // Photon is on the same direction as the detector normal
    //   Q = Pold;

    //   CMatrix Smatrix = current_medium->scattering_matrix(1.0, 0.0);
    //   double s2 = std::norm(Smatrix(0, 0));
    //   double s1 = std::norm(Smatrix(1, 1));

    //   F = (s2 + s1) / 2.0;
    //   Ed1 = E1old * s2 / sqrt(F);
    //   Ed2 = E2old * s1 / sqrt(F);
    // }
    // else if (std::abs(1 + mu) < 1e-11)
    // {
    //   // Photon is on the opposite direction of the detector normal
    //   Q = Pold;
    //   Q(1, 0) *= -1;
    //   Q(1, 1) *= -1;
    //   Q(1, 2) *= -1;
    //   Q(2, 0) *= -1;
    //   Q(2, 1) *= -1;
    //   Q(2, 2) *= -1;

    //   CMatrix Smatrix = current_medium->scattering_matrix(M_PI, 0.0);
    //   double s2 = std::norm(Smatrix(0, 0));
    //   double s1 = std::norm(Smatrix(1, 1));

    //   F = (s2 + s1) / 2.0;
    //   Ed1 = E1old * s2 / sqrt(F);
    //   Ed2 = E2old * s1 / sqrt(F);
    // }
    // else
    // {
    //   // Cross product to find the rotation axis and angle
    //   p = {
    //       (Pold(2, 1) * wd - Pold(2, 2) * vd) / nu,
    //       (Pold(2, 2) * ud - Pold(2, 0) * wd) / nu,
    //       (Pold(2, 0) * vd - Pold(2, 1) * ud) / nu};

    //   // Dot product to find the cosine and sine of the rotation angle
    //   double sinphi = -(Pold(0, 0) * p.x + Pold(0, 1) * p.y + Pold(0, 2) * p.z); // m dot p
    //   double cosphi = Pold(1, 0) * p.x + Pold(1, 1) * p.y + Pold(1, 2) * p.z;    // n dot p

    //   A(0, 0) = mu * cosphi;
    //   A(0, 1) = mu * sinphi;
    //   A(0, 2) = -nu;
    //   A(1, 0) = -sinphi;
    //   A(1, 1) = cosphi;
    //   A(1, 2) = 0;
    //   A(2, 0) = nu * cosphi;
    //   A(2, 1) = nu * sinphi;
    //   A(2, 2) = mu;

    //   matmul(A, Pold, Q);

    //   double theta = std::acos(mu);
    //   CMatrix Smatrix = current_medium->scattering_matrix(theta, 0);
    //   double s2 = std::norm(Smatrix(0, 0));
    //   double s1 = std::norm(Smatrix(1, 1));

    //   double s2sq = std::norm(Smatrix(0, 0));
    //   double s1sq = std::norm(Smatrix(1, 1));

    //   double e1sq = std::norm(E1old);
    //   double e2sq = std::norm(E2old);
    //   double e12 = (E1old * conj(E2old)).real();

    //   F = (s2sq * e1sq + s1sq * e2sq) * cosphi * cosphi + (s1sq * e1sq + s2sq * e2sq) * sinphi * sinphi + 2 * (s2sq - s1sq) * e12 * cosphi * sinphi;
    //   Ed1 = (cosphi * E1old + sinphi * E2old) * s2 / sqrt(F);
    //   Ed2 = (-sinphi * E1old + cosphi * E2old) * s1 / sqrt(F);
    // }

    // double deposit = 0.0;
    // double z = photon.pos.z;
    // double zd = origin.z;
    // double weight = photon.weight;
    // double csca = 1.0;

    // if (photon.events == 0)
    //   /* ballistic light */
    //   if (std::abs(1 - mu) < 1e-11)
    //     deposit = weight * exp(-fabs((z - zd) / wd));
    //   else
    //     deposit = 0;
    // else
    //   deposit = weight * F / csca * exp(-fabs((z - zd) / wd));

    // double t = photon.launch_time + (photon.opticalpath / photon.velocity);
    // double td = t + fabs((z - zd) / wd);

    // std::complex<double> phase = std::exp(std::complex<double>(0, photon.k * td));
    // std::complex<double> Ex = (Ed1 * Q(0, 0) + Ed2 * Q(1, 0)) * phase;
    // std::complex<double> Ey = (Ed1 * Q(0, 1) + Ed2 * Q(1, 1)) * phase;

    // this->Ex(x_idx, y_idx) += Ex * sqrt(deposit);
    // this->Ey(x_idx, y_idx) += Ey * sqrt(deposit);
    // hits += 1;
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
      // Bin 0 is reserved for time-integrated accumulation.
      // Time windows occupy bins [1, N_t-1].
      N_t = static_cast<int>(std::ceil(len_t / dt)) + 1;
    }

    N_x = static_cast<int>(std::ceil(len_x / dx));
    N_y = static_cast<int>(std::ceil(len_y / dy));

    S0.resize(N_t, Matrix(N_x, N_y));
    S1.resize(N_t, Matrix(N_x, N_y));
    S2.resize(N_t, Matrix(N_x, N_y));
    S3.resize(N_t, Matrix(N_x, N_y));

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
    det->filter_direction_enabled = filter_direction_enabled;
    det->filter_direction = filter_direction;
    det->filter_events_enabled = filter_events_enabled;
    det->filter_events_min = filter_events_min;
    det->filter_events_max = filter_events_max;
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
          S0[t](i, j) += o.S0[t](i, j);
          S1[t](i, j) += o.S1[t](i, j);
          S2[t](i, j) += o.S2[t](i, j);
          S3[t](i, j) += o.S3[t](i, j);
        }
      }
    }
  }

  void PlanarFluenceSensor::process_hit(Photon &photon, InteractionInfo &info, const Sample &medium)
  {
    // --- Time-resolved binning ---
    // If dt==0, all photons go into a single time bin (steady-state mode).
    int t_idx = -1;
    if (dt > 0)
    {
      double arrival_time = photon.launch_time + (info.opticalpath_at_hit / photon.velocity);
      if (arrival_time < 0 || arrival_time >= len_t)
        return;
      t_idx = static_cast<int>(arrival_time / dt) + 1;
      if (t_idx >= N_t)
        return;
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

    // If time windows are enabled, accumulate into the corresponding temporal bin.
    if (dt > 0)
    {
      S0[t_idx](x_idx, y_idx) += S0_contribution;
      S1[t_idx](x_idx, y_idx) += S1_contribution;
      S2[t_idx](x_idx, y_idx) += S2_contribution;
      S3[t_idx](x_idx, y_idx) += S3_contribution;
    }

    // Always accumulate in bin 0 for time-integrated fluence.
    S0[0](x_idx, y_idx) += S0_contribution;
    S1[0](x_idx, y_idx) += S1_contribution;
    S2[0](x_idx, y_idx) += S2_contribution;
    S3[0](x_idx, y_idx) += S3_contribution;

    hits += 1;
  }

  // TODO: Implement
  void PlanarFluenceSensor::process_estimation(const Photon &photon, const Sample &medium)
  {
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // FarFieldCBSSensor implementation
  // ═══════════════════════════════════════════════════════════════════════════
  FarFieldCBSSensor::FarFieldCBSSensor(double theta_max, double phi_max, double len_t, double d_theta, double d_phi, double d_t, bool estimator) : Sensor(0.0, true, estimator)
  {
    this->theta_max = theta_max;
    this->phi_max = phi_max;
    this->t_max = len_t;

    dtheta = d_theta;
    dphi = d_phi;
    dt = d_t;

    if (dt == 0)
    {
      N_t = 1;
    }
    else
    {
      // Bin 0 is reserved for time-integrated accumulation.
      // Time windows occupy bins [1, N_t-1].
      N_t = static_cast<int>(std::ceil(t_max / dt)) + 1;
    }

    N_theta = static_cast<int>(std::ceil(theta_max / dtheta));
    N_phi = static_cast<int>(std::ceil(phi_max / dphi));

    S0_coh.resize(N_t, Matrix(N_theta, N_phi));
    S1_coh.resize(N_t, Matrix(N_theta, N_phi));
    S2_coh.resize(N_t, Matrix(N_theta, N_phi));
    S3_coh.resize(N_t, Matrix(N_theta, N_phi));
    S0_incoh.resize(N_t, Matrix(N_theta, N_phi));
    S1_incoh.resize(N_t, Matrix(N_theta, N_phi));
    S2_incoh.resize(N_t, Matrix(N_theta, N_phi));
    S3_incoh.resize(N_t, Matrix(N_theta, N_phi));

    // Precompute the angular bin-center trig and per-θ solid-angle bands once.
    // These are constant across all photons/events and are reused by the
    // estimator's per-bin loop (process_estimation).
    cos_th_det.resize(N_theta);
    sin_th_det.resize(N_theta);
    dOmega_theta.resize(N_theta);
    for (int it = 0; it < N_theta; ++it)
    {
      const double th_det = (it + 0.5) * dtheta;
      cos_th_det[it] = std::cos(th_det);
      sin_th_det[it] = std::sin(th_det);
      dOmega_theta[it] = std::cos(it * dtheta) - std::cos((it + 1) * dtheta);
    }
    cos_ph_det.resize(N_phi);
    sin_ph_det.resize(N_phi);
    phi_values.resize(N_phi);
    for (int jp = 0; jp < N_phi; ++jp)
    {
      const double ph_det = (jp + 0.5) * dphi;
      cos_ph_det[jp] = std::cos(ph_det);
      sin_ph_det[jp] = std::sin(ph_det);
      phi_values[jp] = ph_det;
    }
  }

  void FarFieldCBSSensor::set_phi_slices(const std::vector<double> &phi_angles)
  {
    if (phi_angles.empty())
      throw std::invalid_argument("set_phi_slices: phi_angles must be non-empty");

    std::vector<double> angles = phi_angles;
    for (double a : angles)
      if (a < 0.0 || a > 2.0 * M_PI)
        throw std::invalid_argument("set_phi_slices: every angle must be in [0, 2*pi]");
    std::sort(angles.begin(), angles.end());
    angles.erase(std::unique(angles.begin(), angles.end()), angles.end());

    phi_explicit = true;
    N_phi = static_cast<int>(angles.size());

    cos_ph_det.resize(N_phi);
    sin_ph_det.resize(N_phi);
    phi_values.resize(N_phi);
    for (int jp = 0; jp < N_phi; ++jp)
    {
      cos_ph_det[jp] = std::cos(angles[jp]);
      sin_ph_det[jp] = std::sin(angles[jp]);
      phi_values[jp] = angles[jp];
    }

    // Re-shape the Stokes accumulators to the new φ-column count (zeroed).
    S0_coh.assign(N_t, Matrix(N_theta, N_phi));
    S1_coh.assign(N_t, Matrix(N_theta, N_phi));
    S2_coh.assign(N_t, Matrix(N_theta, N_phi));
    S3_coh.assign(N_t, Matrix(N_theta, N_phi));
    S0_incoh.assign(N_t, Matrix(N_theta, N_phi));
    S1_incoh.assign(N_t, Matrix(N_theta, N_phi));
    S2_incoh.assign(N_t, Matrix(N_theta, N_phi));
    S3_incoh.assign(N_t, Matrix(N_theta, N_phi));
  }

  void FarFieldCBSSensor::merge_from(const Sensor &other)
  {
    const auto &o = dynamic_cast<const FarFieldCBSSensor &>(other);
    hits += o.hits;
    for (auto t = 0; t < N_t; ++t)
    {
      for (int i = 0; i < N_theta; ++i)
      {
        for (int j = 0; j < N_phi; ++j)
        {
          S0_coh[t](i, j) += o.S0_coh[t](i, j);
          S1_coh[t](i, j) += o.S1_coh[t](i, j);
          S2_coh[t](i, j) += o.S2_coh[t](i, j);
          S3_coh[t](i, j) += o.S3_coh[t](i, j);
          S0_incoh[t](i, j) += o.S0_incoh[t](i, j);
          S1_incoh[t](i, j) += o.S1_incoh[t](i, j);
          S2_incoh[t](i, j) += o.S2_incoh[t](i, j);
          S3_incoh[t](i, j) += o.S3_incoh[t](i, j);
        }
      }
    }
  }

  std::unique_ptr<Sensor> FarFieldCBSSensor::clone() const
  {
    auto det = std::make_unique<FarFieldCBSSensor>(theta_max, phi_max, t_max, dtheta, dphi, dt, estimator_enabled);
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
    det->filter_direction_enabled = filter_direction_enabled;
    det->filter_direction = filter_direction;
    det->filter_events_enabled = filter_events_enabled;
    det->filter_events_min = filter_events_min;
    det->filter_events_max = filter_events_max;
    det->theta_pp_max = theta_pp_max;
    // Reproduce the explicit φ-slice grid (idempotent: phi_values is already
    // sorted/unique). Must run so per-worker clones and add_detector's internal
    // copy share the same φ-column count that merge_from() relies on.
    if (phi_explicit)
      det->set_phi_slices(phi_values);
    return det;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // CBS shared helpers (file-local)
  //
  // The reverse-path 3-stage rotation algorithm (Xu 2008, Eq. 3;
  //   E_rev = S^(1) R(φ1') T^rev R(φn') S^(n) R(φn) E0,   T^rev = Q T^T Q)
  // lives in EXACTLY ONE place: reverse_field(). Both the direct detector
  // (process_hit) and the last-flight estimator (process_estimation) call it,
  // differing only in WHICH frames and WHICH raw interior product they pass.
  // Edit the rotations once and both paths stay consistent.
  // ═══════════════════════════════════════════════════════════════════════════

  // Normalization factor F: expected scattered intensity for a given input Jones
  // vector and azimuth. Used identically by transport (run_photon) and by the
  // estimator's angular density p(Ω) = F / (π · I_norm).
  static double phase_F(const CMatrix &S, double cos_phi, double sin_phi, const CVec2 &E)
  {
    const double s22 = std::norm(S(0, 0));
    const double s11 = std::norm(S(1, 1));
    const double Emm = std::norm(E.m);
    const double Enn = std::norm(E.n);
    const double c2 = cos_phi * cos_phi;
    const double s2 = sin_phi * sin_phi;
    return Emm * (s22 * c2 + s11 * s2) +
           Enn * (s22 * s2 + s11 * c2) +
           2.0 * std::real(E.m * std::conj(E.n)) * (s22 - s11) * sin_phi * cos_phi;
  }

  // Build the exit local frame after scattering by (θ, φ), using the SAME
  // A-update convention as the transport loop (rows = m', n', s').
  static Matrix scatter_frame(const Matrix &P, double theta, double cos_phi, double sin_phi)
  {
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    Matrix A(3, 3);
    A(0, 0) = ct * cos_phi;
    A(0, 1) = ct * sin_phi;
    A(0, 2) = -st;
    A(1, 0) = -sin_phi;
    A(1, 1) = cos_phi;
    A(1, 2) = 0;
    A(2, 0) = st * cos_phi;
    A(2, 1) = st * sin_phi;
    A(2, 2) = ct;
    Matrix out(3, 3);
    matmul(A, P, out);
    return out;
  }

  // Project a local Jones vector (m, n) onto the lab (x, y) plane.
  // P rows are (m, n, s); columns 0/1 are the x/y lab components.
  static void project_to_lab(const CVec2 &E, const Matrix &P,
                             std::complex<double> &Ex, std::complex<double> &Ey)
  {
    Ex = E.m * P(0, 0) + E.n * P(1, 0);
    Ey = E.m * P(0, 1) + E.n * P(1, 1);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  //  Reverse-path electric field — the 3-stage time-reversed (reciprocal) algo
  //
  //  Stage A : first reverse scatter at r_n,   s0      -> -s_{n-1}
  //  Stage B : interior segment via reciprocity, E <- Q · T_interior^T · Q · E
  //  Stage C : last reverse scatter at r_1,     -s1     ->  s_n
  //
  //  Uses the RAW (un-F-normalized) interior product and normalizes the result
  //  ONCE at the end (in the suffix). Energy is carried by photon.weight, not by
  //  the Jones norm, so any scalar inside the path cancels — keeping this
  //  consistent with the normalized forward field used by both callers.
  //
  //  The algorithm is split into a PREFIX (Stages A+B) and a SUFFIX (Stage C).
  //  In the CBS estimator, the prefix depends only on P0, P_{n-1} (= current
  //  frame), the interior product and the input polarization — all FIXED across
  //  the angular bins — while only Stage C depends on the exit frame P_n that
  //  varies per bin. Hoisting the prefix out of the bin loop avoids recomputing
  //  Stage A's scattering matrix and the interior reciprocity product for every
  //  bin. The split is exact: prefix∘suffix is bit-identical to the old monolith.
  // ─────────────────────────────────────────────────────────────────────────────

  // Stages A + B. Returns the Jones vector right before the last reverse scatter.
  // `med_n` is the species at the LAST scattering vertex (r_n) — the amplitude
  // matrix for Stage A is taken from it, not from the layer index, so mixtures
  // use the species the photon actually scattered against.
  static CVec2 reverse_field_prefix(const Matrix &P0, const Matrix &P_nm1,
                                    const CMatrix &T_interior_raw,
                                    const ScatteringMedium *med_n, const CVec2 &E_in)
  {
    const Vec3 s0 = row_vec3(P0, 2);
    const Vec3 snm1 = row_vec3(P_nm1, 2);
    const Vec3 m0 = row_vec3(P0, 0), n0 = row_vec3(P0, 1);
    const Vec3 mnm1 = row_vec3(P_nm1, 0), nnm1 = row_vec3(P_nm1, 1);

    CMatrix Q(2, 2);
    Q(0, 0) = 1;
    Q(0, 1) = 0;
    Q(1, 0) = 0;
    Q(1, 1) = -1;

    // ── Stage A: scatter at r_n, s0 -> -s_{n-1} ──
    const Vec3 s_in_a = s0;
    const Vec3 s_out_a = snm1 * (-1.0);
    const double th_a = std::acos(clamp_pm1(dot(s_in_a, s_out_a)));
    const CMatrix S_a = med_n->scattering_matrix(th_a, 0.0);

    const Vec3 nprime = safe_unit(cross(s_in_a, s_out_a), n0);
    const Vec3 mprime_in = safe_unit(cross(nprime, s_in_a), m0);
    const Vec3 mprime_out = safe_unit(cross(nprime, s_out_a), mnm1);

    CMatrix SR_a(2, 2);
    matcmul(S_a, rot2(mprime_in, nprime, m0, n0), SR_a); // S(θ)·R(φn)
    CVec2 E = apply2(SR_a, E_in);
    E = apply2(rot2(mnm1, nnm1 * (-1.0), mprime_out, nprime), E); // R(φn'): -> (m_{n-1}, -n_{n-1})

    // ── Stage B: interior via reciprocity  E <- Q T^T Q E ──
    CMatrix Tt(2, 2);
    Tt(0, 0) = T_interior_raw(0, 0);
    Tt(0, 1) = T_interior_raw(1, 0);
    Tt(1, 0) = T_interior_raw(0, 1);
    Tt(1, 1) = T_interior_raw(1, 1);
    E = apply2(Q, E);
    E = apply2(Tt, E);
    E = apply2(Q, E);

    return E;
  }

  // Stage C + final normalization. `E_prefix` is the output of reverse_field_prefix.
  // `med_1` is the species at the FIRST scattering vertex (r_1).
  static CVec2 reverse_field_suffix(const Matrix &P1, const Matrix &P_n,
                                    const ScatteringMedium *med_1, const CVec2 &E_prefix)
  {
    const Vec3 s1 = row_vec3(P1, 2);
    const Vec3 sn = row_vec3(P_n, 2);
    const Vec3 m1 = row_vec3(P1, 0), n1 = row_vec3(P1, 1);
    const Vec3 mn = row_vec3(P_n, 0), nn = row_vec3(P_n, 1);

    // ── Stage C: scatter at r_1, -s1 -> sn ──
    const Vec3 s_in_c = s1 * (-1.0);
    const Vec3 s_out_c = sn;
    const double th_c = std::acos(clamp_pm1(dot(s_in_c, s_out_c)));
    const CMatrix S_c = med_1->scattering_matrix(th_c, 0.0);

    const Vec3 npp = safe_unit(cross(s_in_c, s_out_c), n1 * (-1.0));
    const Vec3 mpp_in = safe_unit(cross(npp, s_in_c), m1);
    const Vec3 mpp_out = safe_unit(cross(npp, s_out_c), mn);

    CMatrix SR_c(2, 2);
    matcmul(S_c, rot2(mpp_in, npp, m1, n1 * (-1.0)), SR_c); // S(θ)·R(φ1')
    CVec2 E = apply2(SR_c, E_prefix);
    E = apply2(rot2(mn, nn, mpp_out, npp), E); // R_out: -> exit basis (m_n, n_n)

    // Normalize once.
    const double nrm = std::sqrt(std::norm(E.m) + std::norm(E.n));
    if (nrm > 1e-300)
    {
      E.m /= nrm;
      E.n /= nrm;
    }
    else
    {
      E.m = 0;
      E.n = 0;
    }
    return E;
  }

  // Convenience wrapper: full prefix∘suffix. Used by the direct detector
  // (process_hit), which evaluates one exit frame per photon.
  static CVec2 reverse_field(const Matrix &P0, const Matrix &P1,
                             const Matrix &P_nm1, const Matrix &P_n,
                             const CMatrix &T_interior_raw,
                             const ScatteringMedium *med_n, const ScatteringMedium *med_1,
                             const CVec2 &E_in)
  {
    const CVec2 prefix = reverse_field_prefix(P0, P_nm1, T_interior_raw, med_n, E_in);
    return reverse_field_suffix(P1, P_n, med_1, prefix);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  //  Sensor helpers
  // ─────────────────────────────────────────────────────────────────────────────

  // Time bin index:  0  = time-resolved disabled (use bin 0 only)
  //                 >=1 = valid resolved bin
  //                 -1  = out of range, skip this contribution.
  int FarFieldCBSSensor::time_bin(double arrival_time) const
  {
    if (dt <= 0)
      return 0;
    if (arrival_time < 0 || arrival_time >= t_max)
      return -1;
    const int idx = static_cast<int>(arrival_time / dt) + 1;
    return (idx < N_t) ? idx : -1;
  }

  // Accumulate coherent (|E_f + E_r|²) and incoherent (|E_f|² + |E_r|²) Stokes
  // into the time-integrated bin 0 and, if enabled, the resolved bin t_idx.
  void FarFieldCBSSensor::accumulate_stokes(int it, int jp, int t_idx,
                                            std::complex<double> Efx, std::complex<double> Efy,
                                            std::complex<double> Erx, std::complex<double> Ery)
  {
    // Coherent: fields interfere.
    const std::complex<double> Ex = Efx + Erx;
    const std::complex<double> Ey = Efy + Ery;
    const double S0c = std::norm(Ex) + std::norm(Ey);
    const double S1c = std::norm(Ex) - std::norm(Ey);
    const double S2c = 2.0 * std::real(Ex * std::conj(Ey));
    const double S3c = 2.0 * std::imag(Ex * std::conj(Ey));

    // Incoherent: intensities add (no interference) — background baseline.
    const double S0i = std::norm(Efx) + std::norm(Efy) + std::norm(Erx) + std::norm(Ery);
    const double S1i = std::norm(Efx) - std::norm(Efy) + std::norm(Erx) - std::norm(Ery);
    const double S2i = 2.0 * (std::real(Efx * std::conj(Efy)) + std::real(Erx * std::conj(Ery)));
    const double S3i = 2.0 * (std::imag(Efx * std::conj(Efy)) + std::imag(Erx * std::conj(Ery)));

    auto add = [&](int t)
    {
      S0_coh[t](it, jp) += S0c;
      S1_coh[t](it, jp) += S1c;
      S2_coh[t](it, jp) += S2c;
      S3_coh[t](it, jp) += S3c;
      S0_incoh[t](it, jp) += S0i;
      S1_incoh[t](it, jp) += S1i;
      S2_incoh[t](it, jp) += S2i;
      S3_incoh[t](it, jp) += S3i;
    };
    add(0);
    if (t_idx >= 1)
      add(t_idx);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  //  Direct detection
  // ─────────────────────────────────────────────────────────────────────────────
  void FarFieldCBSSensor::process_hit(Photon &photon, InteractionInfo &info, const Sample &medium)
  {
    if (estimator_enabled)
      return;

    // CBS needs ≥2 events to form a time-reversed pair.
    if (photon.events < 2)
    {
      const int t_idx = time_bin(photon.launch_time + info.opticalpath_at_hit / photon.velocity);
      if (t_idx < 0)
        return;

      const Matrix &P = photon.P_local;
      const Vec3 s_out{P(2, 0), P(2, 1), P(2, 2)};
      const double theta = std::acos(-s_out.z);
      double phi = std::atan2(s_out.y, s_out.x);
      if (phi < 0)
        phi += 2.0 * M_PI;
      const int it = static_cast<int>(theta / dtheta);
      const int jp = static_cast<int>(phi / dphi);
      if (it < 0 || it >= N_theta || jp < 0 || jp >= N_phi)
        return;

      const double w_sqrt = std::sqrt(photon.weight);
      const std::complex<double> Em = photon.polarization.m * info.phase * w_sqrt;
      const std::complex<double> En = photon.polarization.n * info.phase * w_sqrt;
      const std::complex<double> E_det_x = (Em * P(0, 0) + En * P(1, 0)) * -1.0;
      const std::complex<double> E_det_y = (Em * P(0, 1) + En * P(1, 1));

      // Single scatter = su propio reverso: deposita |E_f|^2 idéntico en coh e inc
      // (fondo sin realce). Er = 0 → accumulate_stokes hace coh = inc.
      accumulate_stokes(it, jp, t_idx, E_det_x, E_det_y, 0.0, 0.0);
      hits += 1;
      return;
    }

    const int t_idx = time_bin(photon.launch_time + info.opticalpath_at_hit / photon.velocity);
    if (t_idx < 0)
      return;

    // Far-field angular bin (θ=0 is exact backscatter, the -z direction).
    const Matrix &P = photon.P_local;
    const Vec3 s_out{P(2, 0), P(2, 1), P(2, 2)};
    const double theta = std::acos(-s_out.z);
    double phi = std::atan2(s_out.y, s_out.x);
    if (phi < 0)
      phi += 2.0 * M_PI;
    const int it = static_cast<int>(theta / dtheta);
    const int jp = static_cast<int>(phi / dphi);
    if (it < 0 || it >= N_theta || jp < 0 || jp >= N_phi)
      return;

    // Reverse-path Jones vector in the exit basis (m_n, n_n).
    const CVec2 Er_loc = reverse_field(
        photon.P0, photon.P1, photon.Pn1, photon.Pn,
        photon.matrix_T_raw, photon.last_scatter_medium, photon.first_scatter_medium,
        photon.initial_polarization);
    photon.polarization_reverse = Er_loc;

    // CBS geometric phase: exp(i (s_out + s_in)·k · (r_n - r_1)).
    const Vec3 s_in{photon.P0(2, 0), photon.P0(2, 1), photon.P0(2, 2)};
    const Vec3 qb = (s_out + s_in) * photon.k;
    const std::complex<double> path_phase =
        std::exp(std::complex<double>(0.0, dot(qb, photon.r_n - photon.r_1)));

    // Common amplitude (info.phase cancels in the coherent term but keeps the
    // absolute phase consistent between forward and reverse).
    const std::complex<double> amp = info.phase * std::sqrt(photon.weight);

    // Forward is tracked continuously; reverse carries the CBS phase. Both in P.
    const CVec2 Ef{photon.polarization.m * amp, photon.polarization.n * amp};
    const CVec2 Er{Er_loc.m * amp * path_phase, Er_loc.n * amp * path_phase};

    std::complex<double> Efx, Efy, Erx, Ery;
    project_to_lab(Ef, P, Efx, Efy);
    project_to_lab(Er, P, Erx, Ery);

    accumulate_stokes(it, jp, t_idx, Efx, Efy, Erx, Ery);
    hits += 1;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  //  Last-flight estimation (forced detection over every angular bin)
  // ─────────────────────────────────────────────────────────────────────────────
  void FarFieldCBSSensor::process_estimation(const Photon &photon, const Sample &medium)
  {
    // Scattering-order filter. process_estimation does NOT go through
    // check_conditions(), so set_events_limit() would otherwise be ignored here,
    // breaking the order-by-order study. We filter on the committed event count
    // (photon.events); note the estimated path actually has events+1 scatters,
    // so to match a given order N of the direct detector set the limit to N-1.
    if (filter_events_enabled &&
        (static_cast<int>(photon.events) < filter_events_min ||
         static_cast<int>(photon.events) > filter_events_max))
      return;

    // Two distinct roles at this vertex:
    //   - `med` is the SPECIES selected for this scatter (active_medium): it
    //     drives S(θ), F, and the I_norm angular normalization.
    //   - `scatter_layer` provides the layer's AGGREGATE albedo (μ_s/μ_t) for
    //     the weight. For a HomogeneousLayer the two coincide.
    const ScatteringMedium *med = photon.active_medium;
    const Layer &scatter_layer = medium.get_layer(photon.current_layer);

    // I_norm cache, one slot per species pointer (keyed implicitly on the medium;
    // k is fixed within a run and acts only as an invalidation guard). With a
    // mixture `med` changes every event, so a single slot would recompute the
    // 2048-point integral each time.
    if (std::abs(_I_norm_k - photon.k) > 1e-12)
    {
      _I_norm_by_medium.clear(); // k changed: every cached value is stale.
      _I_norm_k = photon.k;
    }
    double _I_norm;
    {
      auto it = _I_norm_by_medium.find(med);
      if (it != _I_norm_by_medium.end())
        _I_norm = it->second;
      else
      {
        _I_norm = compute_I_norm(*med, photon.k);
        _I_norm_by_medium.emplace(med, _I_norm);
      }
    }
    if (_I_norm < 1e-300)
      return;

    // ── Single-scatter branch (events == 0) ──────────────────────────────────
    // The forced scatter is the first AND last scatter of a 1-scatter path.
    // Single scatter is its own time-reverse: there is no reverse field, and the
    // CBS geometric phase vanishes (r_n = r_1 ⇒ q·(r_n−r_1)=0). Deposit |E_f|²
    // identically in coherent and incoherent (unenhanced background), mirroring
    // the events<2 branch of process_hit so both detection modes agree.
    if (photon.events == 0)
    {
      // Detector grid around the backscatter axis −s_in. P_local == P0 here
      // (no scatter rotation applied yet), so s_cur is the incidence direction.
      const Vec3 e1 = row_vec3(photon.P0, 0);
      const Vec3 e2 = row_vec3(photon.P0, 1);
      const Vec3 s_in = row_vec3(photon.P0, 2);
      const Vec3 e3 = s_in * (-1.0);

      const Matrix &Pcur = photon.P_local;
      const Vec3 m_cur = row_vec3(Pcur, 0);
      const Vec3 n_cur = row_vec3(Pcur, 1);
      const Vec3 s_cur = row_vec3(Pcur, 2); // == s_in

      const double w_scatter = photon.weight * (scatter_layer.mu_scattering() / scatter_layer.mu_attenuation());
      if (w_scatter < 1e-300)
        return;

      const double th_cap = (theta_pp_max > 0.0) ? std::min(theta_pp_max, theta_max) : theta_max;
      const int i_max = std::min(N_theta, static_cast<int>(th_cap / dtheta));

      for (int it = 0; it < i_max; ++it)
      {
        const double dOmega_band = dOmega_theta[it];
        const double s_th = sin_th_det[it];
        const double c_th = cos_th_det[it];

        for (int jp = 0; jp < N_phi; ++jp)
        {
          const double c_ph = cos_ph_det[jp];
          const double s_ph = sin_ph_det[jp];

          const Vec3 s_out = e1 * (s_th * c_ph) + e2 * (s_th * s_ph) + e3 * c_th;

          double L;
          if (!intersect_plane(photon.pos, s_out, origin, normal, L))
            continue;
          const double Tr = std::exp(-optical_depth_to_detector(medium, photon.pos, s_out, L));
          if (Tr < 1e-20)
            continue;

          const Vec3 pv = cross(s_cur, s_out);
          const double pn = norm(pv);
          if (pn < 1e-12)
            continue; // forward/backward degenerate scattering plane
          const Vec3 p_hat = pv * (1.0 / pn);
          const double sin_phi = -dot(m_cur, p_hat);
          const double cos_phi = dot(n_cur, p_hat);
          const double th_scat = std::acos(clamp_pm1(dot(s_cur, s_out)));

          const CMatrix S = med->scattering_matrix(th_scat, 0.0);

          const double F = phase_F(S, cos_phi, sin_phi, photon.polarization);
          if (F < 1e-300)
            continue;
          const double prob_bin = (F / (M_PI * _I_norm)) * (dOmega_band * dphi);
          const double w_bin = w_scatter * Tr * prob_bin;
          if (w_bin < 1e-300)
            continue;

          const std::complex<double> amp =
              std::sqrt(w_bin) * std::exp(std::complex<double>(0.0, photon.k * L));

          const Matrix P_exit = scatter_frame(Pcur, th_scat, cos_phi, sin_phi);

          // Forward field only — no reverse, no CBS path phase.
          const CVec2 Ef_loc = apply_scatter_normalized(S, cos_phi, sin_phi, photon.polarization);

          const int t_idx = time_bin(photon.launch_time + (photon.opticalpath + L) / photon.velocity);
          if (t_idx < 0)
            continue;

          const CVec2 Ef{Ef_loc.m * amp, Ef_loc.n * amp};

          std::complex<double> Efx, Efy;
          project_to_lab(Ef, P_exit, Efx, Efy);

          // Er = 0 ⇒ accumulate_stokes deposits coherent == incoherent.
          accumulate_stokes(it, jp, t_idx, Efx, Efy, 0.0, 0.0);
        }
      }
      hits += 1;
      return;
    }

    // ── Multiple-scatter branch (events >= 1): estimated path = events+1 ≥ 2 ──
    // Detector angular grid is built around the backscatter axis -s_in.
    const Vec3 e1 = row_vec3(photon.P0, 0);
    const Vec3 e2 = row_vec3(photon.P0, 1);
    const Vec3 s_in = row_vec3(photon.P0, 2);
    const Vec3 e3 = s_in * (-1.0); // θ_det = 0 → exact backscatter

    const Matrix &Pcur = photon.P_local;
    const Vec3 m_cur = row_vec3(Pcur, 0);
    const Vec3 n_cur = row_vec3(Pcur, 1);
    const Vec3 s_cur = row_vec3(Pcur, 2);

    // Raw interior product for the estimated path (n = events+1): the scatter
    // just executed becomes the new penultimate (interior) event, so we fold the
    // buffered J into the committed product.
    CMatrix Tmid_raw = photon.matrix_T_raw;
    if (photon.events >= 2 && photon.has_T_prev)
    {
      CMatrix tmp(2, 2);
      matcmul(photon.matrix_T_raw_buffer, photon.matrix_T_raw, tmp);
      Tmid_raw = std::move(tmp);
    }

    const double w_scatter = photon.weight * (med->mu_scattering / med->mu_attenuation);
    if (w_scatter < 1e-300)
      return;

    const double th_cap = (theta_pp_max > 0.0) ? std::min(theta_pp_max, theta_max) : theta_max;
    const int i_max = std::min(N_theta, static_cast<int>(th_cap / dtheta));

    // Reverse-path PREFIX (Stages A+B) is identical for every angular bin: it
    // depends only on P0, the current frame Pcur (= penultimate scatter), the
    // interior product Tmid_raw and the input polarization — none of which vary
    // across bins. Compute it once here; the per-bin loop only runs Stage C.
    const CVec2 rev_prefix = reverse_field_prefix(
        photon.P0, Pcur, Tmid_raw, photon.active_medium, photon.initial_polarization);

    for (int it = 0; it < i_max; ++it)
    {
      const double dOmega_band = dOmega_theta[it]; // exact θ-band solid angle (×dφ)
      const double s_th = sin_th_det[it];
      const double c_th = cos_th_det[it];

      for (int jp = 0; jp < N_phi; ++jp)
      {
        const double c_ph = cos_ph_det[jp];
        const double s_ph = sin_ph_det[jp];

        // Bin direction in lab coords: s_out = sinθ cosφ e1 + sinθ sinφ e2 + cosθ e3.
        const Vec3 s_out = e1 * (s_th * c_ph) + e2 * (s_th * s_ph) + e3 * c_th;

        // Path to the detector plane and transmittance without further scatter.
        // The straight ray may cross several layers, so the optical depth is
        // summed layer by layer (Σ μ_{t,i}·L_i) instead of using only the
        // current layer's μ_t.
        double L;
        if (!intersect_plane(photon.pos, s_out, origin, normal, L))
          continue;
        const double Tr = std::exp(-optical_depth_to_detector(medium, photon.pos, s_out, L));
        if (Tr < 1e-20)
          continue;

        // Estimated scatter geometry s_cur -> s_out, expressed in the (m_cur,n_cur) basis.
        const Vec3 pv = cross(s_cur, s_out);
        const double pn = norm(pv);
        if (pn < 1e-12)
          continue; // forward/backward degenerate scattering plane
        const Vec3 p_hat = pv * (1.0 / pn);
        const double sin_phi = -dot(m_cur, p_hat);
        const double cos_phi = dot(n_cur, p_hat);
        const double th_scat = std::acos(clamp_pm1(dot(s_cur, s_out)));

        const CMatrix S = med->scattering_matrix(th_scat, 0.0);

        // Angular density p(Ω) = F / (π I_norm); weight that lands in this bin.
        const double F = phase_F(S, cos_phi, sin_phi, photon.polarization);
        if (F < 1e-300)
          continue;
        const double prob_bin = (F / (M_PI * _I_norm)) * (dOmega_band * dphi);
        const double w_bin = w_scatter * Tr * prob_bin;
        if (w_bin < 1e-300)
          continue;

        const std::complex<double> amp =
            std::sqrt(w_bin) * std::exp(std::complex<double>(0.0, photon.k * L));

        // Estimated exit frame (same convention as transport).
        const Matrix P_exit = scatter_frame(Pcur, th_scat, cos_phi, sin_phi);

        // Forward (unit) and reverse (unit) Jones vectors in the P_exit basis.
        // The reverse prefix (Stages A+B) was hoisted out of the loop; only the
        // bin-dependent Stage C (suffix) runs here.
        const CVec2 Ef_loc = apply_scatter_normalized(S, cos_phi, sin_phi, photon.polarization);
        const CVec2 Er_loc = reverse_field_suffix(
            photon.P1, P_exit, photon.first_scatter_medium, rev_prefix);

        // CBS geometric phase (estimated r_n = current position).
        const Vec3 qb = (s_out + s_in) * photon.k;
        const std::complex<double> path_phase =
            std::exp(std::complex<double>(0.0, dot(qb, photon.pos - photon.r_1)));

        // Time bin uses the extra straight path L to the detector.
        const int t_idx = time_bin(photon.launch_time + (photon.opticalpath + L) / photon.velocity);
        if (t_idx < 0)
          continue;

        const CVec2 Ef{Ef_loc.m * amp, Ef_loc.n * amp};
        const CVec2 Er{Er_loc.m * amp * path_phase, Er_loc.n * amp * path_phase};

        std::complex<double> Efx, Efy, Erx, Ery;
        project_to_lab(Ef, P_exit, Efx, Efy);
        project_to_lab(Er, P_exit, Erx, Ery);

        accumulate_stokes(it, jp, t_idx, Efx, Efy, Erx, Ery);
      }
    }
    hits += 1;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // StatisticsSensor implementation
  // ═══════════════════════════════════════════════════════════════════════════
  StatisticsSensor::StatisticsSensor(double z, bool absorb) : Sensor(z, absorb, false)
  {
    events_histogram = std::vector<std::vector<int>>();
    theta_histogram = std::vector<std::vector<int>>();
    phi_histogram = std::vector<std::vector<int>>();
    depth_histogram = std::vector<std::vector<int>>();
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
    det->filter_direction_enabled = filter_direction_enabled;
    det->filter_direction = filter_direction;
    det->filter_events_enabled = filter_events_enabled;
    det->filter_events_min = filter_events_min;
    det->filter_events_max = filter_events_max;

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
    det->h_max_time = h_max_time;
    det->n_bins_time = n_bins_time;
    det->h_dtime = h_dtime;
    det->weight_histogram_bins_set = weight_histogram_bins_set;
    det->max_weight = max_weight;
    det->n_bins_weight = n_bins_weight;
    det->dweight = dweight;

    det->N_t = N_t;
    det->dt = dt;
    det->t_max = t_max;

    if (events_histogram_bins_set)
    {
      det->events_histogram.assign(N_t, std::vector<int>(max_events, 0));
    }
    if (theta_histogram_bins_set)
    {
      det->theta_histogram.assign(N_t, std::vector<int>(n_bins_theta, 0));
    }
    if (phi_histogram_bins_set)
    {
      det->phi_histogram.assign(N_t, std::vector<int>(n_bins_phi, 0));
    }
    if (depth_histogram_bins_set)
    {
      det->depth_histogram.assign(N_t, std::vector<int>(n_bins_depth, 0));
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
      for (size_t t = 0; t < events_histogram.size(); ++t)
      {
        for (size_t i = 0; i < events_histogram[t].size(); ++i)
        {
          events_histogram[t][i] += o.events_histogram[t][i];
        }
      }
    }
    if (theta_histogram_bins_set && o.theta_histogram_bins_set)
    {
      for (size_t t = 0; t < theta_histogram.size(); ++t)
      {
        for (size_t i = 0; i < theta_histogram[t].size(); ++i)
        {
          theta_histogram[t][i] += o.theta_histogram[t][i];
        }
      }
    }
    if (phi_histogram_bins_set && o.phi_histogram_bins_set)
    {
      for (size_t t = 0; t < phi_histogram.size(); ++t)
      {
        for (size_t i = 0; i < phi_histogram[t].size(); ++i)
        {
          phi_histogram[t][i] += o.phi_histogram[t][i];
        }
      }
    }
    if (depth_histogram_bins_set && o.depth_histogram_bins_set)
    {
      for (size_t t = 0; t < depth_histogram.size(); ++t)
      {
        for (size_t i = 0; i < depth_histogram[t].size(); ++i)
        {
          depth_histogram[t][i] += o.depth_histogram[t][i];
        }
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

  void StatisticsSensor::process_hit(Photon &photon, InteractionInfo &info, const Sample &medium)
  {
    const double arrival_time = photon.launch_time + (info.opticalpath_at_hit / photon.velocity);
    int t_idx = -1;
    if (N_t > 1 && dt > 0.0 && arrival_time >= 0.0 && arrival_time < t_max)
    {
      t_idx = static_cast<int>(arrival_time / dt) + 1;
      if (t_idx >= N_t)
        t_idx = -1;
    }

    auto deposit_temporal = [t_idx](std::vector<std::vector<int>> &hist, int bin_idx, int t_id)
    {
      if (hist.empty())
        return;

      hist[0][bin_idx]++;

      if (t_id >= 1 && t_id < static_cast<int>(hist.size()))
      {
        hist[t_id][bin_idx]++;
      }
    };

    if (events_histogram_bins_set)
    {
      int events = photon.events;
      if (events >= 0 && events < max_events)
      {
        deposit_temporal(events_histogram, events, t_idx);
      }
    }
    if (theta_histogram_bins_set)
    {
      double theta = std::acos(-photon.P_local(2, 2));
      if (theta >= min_theta && theta < max_theta)
      {
        int idx = static_cast<int>((theta - min_theta) / dtheta);
        deposit_temporal(theta_histogram, idx, t_idx);
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
        deposit_temporal(phi_histogram, idx, t_idx);
      }
    }
    if (depth_histogram_bins_set)
    {
      double depth = photon.penetration_depth;
      if (depth >= 0 && depth < max_depth)
      {
        int idx = static_cast<int>(depth / ddepth);
        deposit_temporal(depth_histogram, idx, t_idx);
      }
    }
    if (time_histogram_bins_set)
    {
      int idx = static_cast<int>(arrival_time / h_dtime);
      if (idx >= 1 && idx < n_bins_time)
      {
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

  void StatisticsSensor::set_time_resolution(double len_t, double dt)
  {
    t_max = len_t;
    this->dt = dt;

    if (dt <= 0.0)
    {
      N_t = 1;
      this->dt = 0.0;
    }
    else
    {
      // Bin 0 is integrated; temporal windows are in [1, n_bins_time-1].
      N_t = static_cast<int>(std::ceil(len_t / dt)) + 1;
    }

    if (events_histogram_bins_set)
    {
      events_histogram.assign(N_t, std::vector<int>(max_events, 0));
    }
    if (theta_histogram_bins_set)
    {
      theta_histogram.assign(N_t, std::vector<int>(n_bins_theta, 0));
    }
    if (phi_histogram_bins_set)
    {
      phi_histogram.assign(N_t, std::vector<int>(n_bins_phi, 0));
    }
    if (depth_histogram_bins_set)
    {
      depth_histogram.assign(N_t, std::vector<int>(n_bins_depth, 0));
    }
  }

  void StatisticsSensor::set_events_histogram_bins(int max_events)
  {
    this->max_events = max_events;
    events_histogram_bins_set = true;
    events_histogram.assign(N_t, std::vector<int>(max_events, 0));
  }

  void StatisticsSensor::set_theta_histogram_bins(double min_theta, double max_theta, int n_bins)
  {
    this->min_theta = min_theta;
    this->max_theta = max_theta;
    this->n_bins_theta = n_bins;
    dtheta = (max_theta - min_theta) / n_bins;
    theta_histogram_bins_set = true;
    theta_histogram.assign(N_t, std::vector<int>(n_bins, 0));
  }

  void StatisticsSensor::set_phi_histogram_bins(double min_phi, double max_phi, int n_bins)
  {
    this->min_phi = min_phi;
    this->max_phi = max_phi;
    this->n_bins_phi = n_bins;
    dphi = (max_phi - min_phi) / n_bins;
    phi_histogram_bins_set = true;
    phi_histogram.assign(N_t, std::vector<int>(n_bins, 0));
  }

  void StatisticsSensor::set_depth_histogram_bins(double max_depth, int n_bins)
  {
    this->max_depth = max_depth;
    this->n_bins_depth = n_bins;
    ddepth = max_depth / n_bins;
    depth_histogram_bins_set = true;
    depth_histogram.assign(N_t, std::vector<int>(n_bins, 0));
  }

  void StatisticsSensor::set_time_histogram_bins(double max_time, int n_bins)
  {
    this->h_max_time = max_time;
    this->n_bins_time = std::max(1, n_bins);
    this->h_dtime = max_time / n_bins;
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
