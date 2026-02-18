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
  // SensorsGroup implementation
  void SensorsGroup::add_detector(std::unique_ptr<Sensor> detector)
  {
    u_int new_id = static_cast<u_int>(detectors.size());
    detector->id = new_id;
    double z = detector->origin.z;

    if (detector->estimator_enabled)
    {
      active_estimators.push_back(detector.get());
    }

    z_layers[z].push_back(detector.get());
    detectors.push_back(std::move(detector));
  }

  bool SensorsGroup::record_hit(Photon &photon, const Medium &medium)
  {
    bool photon_killed = false;

    double z1 = photon.prev_pos.z;
    double z2 = photon.pos.z;
    if (std::abs(z2 - z1) < 1e-12)
      return false;

    double z_min = std::min(z1, z2);
    double z_max = std::max(z1, z2);

    auto it_start = z_layers.lower_bound(z_min);
    auto it_end = z_layers.upper_bound(z_max);

    for (auto it = it_start; it != it_end; ++it)
    {
      double z_plane = it->first;
      bool crosses = (z_plane >= z_min) && (z_plane <= z_max);

      if (crosses)
      {
        const Vec3 xn = photon.prev_pos;
        const Vec3 xf = photon.pos;
        const Vec3 d = xf - xn;
        const Vec3 detector_normal{0, 0, 1};
        const Vec3 detector_origin{0, 0, z_plane};

        const double denom = dot(d, detector_normal);
        const double t = dot(detector_origin - xn, detector_normal) / denom;

        const Vec3 hit_point = xn + d * t;
        const double correction_distance = luminis::math::norm(hit_point - xf);
        double opticalpath_correction = photon.opticalpath;
        if (correction_distance > 0)
        {
          opticalpath_correction -= correction_distance;
        }

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
    for (Sensor *sensor : active_estimators)
    {
      sensor->process_estimation(photon, medium);
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

  // Sensor implementation
  Sensor::Sensor(double z)
  {
    origin = {0, 0, z};
    normal = Z_UNIT_VEC3;
    backward_normal = Z_UNIT_VEC3 * -1;
    m_polarization = X_UNIT_VEC3;
    n_polarization = Y_UNIT_VEC3;
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
    if (filter_theta_enabled)
    {
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

  // PhotonRecordSensor implementation
  PhotonRecordSensor::PhotonRecordSensor(double z) : Sensor(z)
  {
  }

  std::unique_ptr<Sensor> PhotonRecordSensor::clone() const
  {
    auto det = std::make_unique<PhotonRecordSensor>(origin.z);
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

  // PlanarFieldSensor implementation
  PlanarFieldSensor::PlanarFieldSensor(double z, double len_x, double len_y, double dx, double dy) : Sensor(z)
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
    auto det = std::make_unique<PlanarFieldSensor>(origin.z, len_x, len_y, dx, dy);
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
    const double x = info.intersection_point.x;
    const double y = info.intersection_point.y;
    const int x_idx = static_cast<int>((x + 0.5 * len_x) / dx);
    const int y_idx = static_cast<int>((y + 0.5 * len_y) / dy);

    if (x_idx < 0 || x_idx >= N_x || y_idx < 0 || y_idx >= N_y)
      return;

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

  // PlanarFluenceSensor implementation
  PlanarFluenceSensor::PlanarFluenceSensor(double z, double len_x, double len_y, double len_t, double dx, double dy, double dt) : Sensor(z)
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
    auto det = std::make_unique<PlanarFluenceSensor>(origin.z, len_x, len_y, len_t, dx, dy, dt);
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

    const double x = info.intersection_point.x;
    const double y = info.intersection_point.y;
    const int x_idx = static_cast<int>((x + 0.5 * len_x) / dx);
    const int y_idx = static_cast<int>((y + 0.5 * len_y) / dy);
    if (x_idx < 0 || x_idx >= N_x || y_idx < 0 || y_idx >= N_y)
      return;

    const double w_sqrt = std::sqrt(photon.weight);
    const std::complex<double> Em_local_photon = photon.polarization.m * info.phase * w_sqrt;
    const std::complex<double> En_local_photon = photon.polarization.n * info.phase * w_sqrt;

    Matrix P = photon.P_local;
    const std::complex<double> E_det_x = (Em_local_photon * P(0, 0) + En_local_photon * P(1, 0)) * -1.0;
    const std::complex<double> E_det_y = (Em_local_photon * P(0, 1) + En_local_photon * P(1, 1));

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

    S0_t[0](x_idx, y_idx) += S0_contribution * deposit;
    S1_t[0](x_idx, y_idx) += S1_contribution * deposit;
    S2_t[0](x_idx, y_idx) += S2_contribution * deposit;
    S3_t[0](x_idx, y_idx) += S3_contribution * deposit;

    hits += 1;
  }

  // PlanarCBSSensor implementation
  PlanarCBSSensor::PlanarCBSSensor(double len_x, double len_y, double dx, double dy) : Sensor(0.0)
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

  // FarFieldFluenceSensor implementation
  FarFieldFluenceSensor::FarFieldFluenceSensor(double z, double theta_max, double phi_max, int n_theta, int n_phi) : Sensor(z)
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
    auto det = std::make_unique<FarFieldFluenceSensor>(origin.z, theta_max, phi_max, N_theta, N_phi);
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

  // FarFieldCBSSensor implementation
  FarFieldCBSSensor::FarFieldCBSSensor(double theta_max, double phi_max, int n_theta, int n_phi) : Sensor(0.0)
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
    auto det = std::make_unique<FarFieldCBSSensor>(theta_max, phi_max, N_theta, N_phi);
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

  void FarFieldCBSSensor::process_hit(Photon &photon, InteractionInfo &info, const Medium &medium)
  {
    if (photon.events < 2)
      return;

    coherent_calculation(photon, medium);

    // Base local final (m,n,s) en el punto de detección
    const Matrix &P = photon.P_local;

    // Dirección de salida (s)
    const Vec3 s_out{P(2, 0), P(2, 1), P(2, 2)};

    // Ángulos far-field (tu convención: theta=0 es backscattering hacia -z)
    const double theta = std::acos(-s_out.z);
    double phi = std::atan2(s_out.y, s_out.x);
    if (phi < 0)
      phi += 2.0 * M_PI;

    const int theta_idx = static_cast<int>(theta / dtheta);
    const int phi_idx = static_cast<int>(phi / dphi);
    if (theta_idx < 0 || theta_idx >= N_theta || phi_idx < 0 || phi_idx >= N_phi)
      return;

    // Fase geométrica CBS: exp(i k (s_out + s_in)·(r_n - r_0))
    const Vec3 s_in{photon.P0(2, 0), photon.P0(2, 1), photon.P0(2, 2)}; // si ya migraste a matrices
    const Vec3 qb = (s_out + s_in) * photon.k;
    const Vec3 delta_r = photon.r_n - photon.r_0;
    const std::complex<double> path_phase = std::exp(std::complex<double>(0, dot(qb, delta_r)));

    // Factor común del trayecto al detector
    const double w_sqrt = std::sqrt(photon.weight);
    const std::complex<double> amp = info.phase * w_sqrt;

    // -------- Forward field (ya está en base local final) ----------
    const std::complex<double> Em_f = photon.polarization.m * amp;
    const std::complex<double> En_f = photon.polarization.n * amp;

    // Proyección a lab (x,y)
    const std::complex<double> Efx = Em_f * P(0, 0) + En_f * P(1, 0) * -1.0;
    const std::complex<double> Efy = Em_f * P(0, 1) + En_f * P(1, 1);

    // -------- Reverse field (usa polarization_reverse) -------------
    const std::complex<double> Em_r = photon.polarization_reverse.m * amp * path_phase;
    const std::complex<double> En_r = photon.polarization_reverse.n * amp * path_phase;

    const std::complex<double> Erx = Em_r * P(0, 0) + En_r * P(1, 0) * -1.0;
    const std::complex<double> Ery = Em_r * P(0, 1) + En_r * P(1, 1);

    // Coherente: |E_f + E_r|^2
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

    // Incoherente: |E_f|^2 + |E_r|^2
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
    if (photon.events < 2)
      return;

    // const double EPSILON = 1e-10;
    // Vec3 r_scat = photon.pos;
    // double z = r_scat.z;
    // double zd = origin.z;

    // Matrix Pold = photon.P_local;
    // Matrix Q = Matrix(3, 3);
    // Matrix A = Matrix(3, 3);

    // std::complex<double> E1old = photon.polarization.m;
    // std::complex<double> E2old = photon.polarization.n;
    // std::complex<double> Ed1;
    // std::complex<double> Ed2;

    // double F;
    // Vec3 current_dir = Vec3{photon.P_local(2, 0), photon.P_local(2, 1), photon.P_local(2, 2)};

    // for (int i = 0; i < N_theta; ++i)
    // {
    //   for (int j = 0; j < N_phi; ++j)
    //   {
    //     double theta = (i + 0.5) * dtheta;
    //     double phi = (j + 0.5) * dphi;

    //     Vec3 S_detector = {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), -std::cos(theta)};

    //     // Distancia proyectada hacia Z=0
    //     double dist_to_boundary = z - zd;
    //     double attenuation = std::exp(-medium.mu_attenuation * dist_to_boundary);
    //     if (attenuation < 1e-12)
    //       continue;

    //     // 4. Geometría de Scattering (Igual que en el espacial, pero con s_out fijo por el loop)
    //     // mu_scat = cos(angulo entre direccion actual y salida)
    //     double mu_scat = dot(current_dir, S_detector);
    //     if (mu_scat > 1.0)
    //       mu_scat = 1.0;
    //     if (mu_scat < -1.0)
    //       mu_scat = -1.0;
    //     double nu_scat = sqrt(1 - mu_scat * mu_scat);

    //     // Vector perpendicular al plano de scattering
    //     Vec3 p = cross(current_dir, S_detector);

    //     // Proyecciones (cosphi, sinphi) del campo incidente sobre el plano
    //     double sinphi = -(Pold(0, 0) * p.x + Pold(0, 1) * p.y + Pold(0, 2) * p.z); // m dot p
    //     double cosphi = Pold(1, 0) * p.x + Pold(1, 1) * p.y + Pold(1, 2) * p.z;    // n dot p

    //     A(0, 0) = mu_scat * cosphi;
    //     A(0, 1) = mu_scat * sinphi;
    //     A(0, 2) = -nu_scat;
    //     A(1, 0) = -sinphi;
    //     A(1, 1) = cosphi;
    //     A(1, 2) = 0;
    //     A(2, 0) = nu_scat * cosphi;
    //     A(2, 1) = nu_scat * sinphi;
    //     A(2, 2) = mu_scat;

    //     matmul(A, Pold, Q);

    //     CMatrix Smatrix = medium.scattering_matrix(theta, 0, photon.k);
    //     double s2 = std::norm(Smatrix(0, 0));
    //     double s1 = std::norm(Smatrix(1, 1));

    //     double s2sq = std::norm(Smatrix(0, 0));
    //     double s1sq = std::norm(Smatrix(1, 1));

    //     double e1sq = std::norm(E1old);
    //     double e2sq = std::norm(E2old);
    //     double e12 = (E1old * conj(E2old)).real();

    //     F = (s2sq * e1sq + s1sq * e2sq) * cosphi * cosphi + (s1sq * e1sq + s2sq * e2sq) * sinphi * sinphi + 2 * (s2sq - s1sq) * e12 * cosphi * sinphi;
    //     Ed1 = (cosphi * E1old + sinphi * E2old) * s2 / sqrt(F);
    //     Ed2 = (-sinphi * E1old + cosphi * E2old) * s1 / sqrt(F);

    //     double deposit = 0.0;
    //     double z = photon.pos.z;
    //     double zd = origin.z;
    //     double weight = photon.weight;
    //     double csca = 1.0;

    //     if (photon.events == 0)
    //       if (std::abs(1 - mu_scat) < 1e-11)
    //         deposit = weight * exp(-fabs((z - zd)));
    //       else
    //         deposit = 0;
    //     else
    //       deposit = weight * F / csca * exp(-fabs((z - zd)));

    //     double t = photon.launch_time + (photon.opticalpath / photon.velocity);
    //     double td = t + fabs((z - zd));

    //     CVec2 E_fwd_local = {Ed1, Ed2};
    //     CVec2 E_rev_local = coherent_estimation(photon, medium, Q);

    //     // Fase geométrica CBS: exp(i k (s_out + s_in)·(r_n - r_0))
    //     const Vec3 s_in{photon.P0(2, 0), photon.P0(2, 1), photon.P0(2, 2)}; // si ya migraste a matrices
    //     const Vec3 qb = (S_detector + s_in) * photon.k;
    //     const Vec3 delta_r = photon.r_n - photon.r_0;
    //     const std::complex<double> path_phase = std::exp(std::complex<double>(0, dot(qb, delta_r)));

    //     // Factor común del trayecto al detector
    //     const double w_sqrt = std::sqrt(photon.weight);
    //     const std::complex<double> amp = std::exp(std::complex<double>(0, photon.k * td)) * w_sqrt;

    //     // -------- Forward field (ya está en base local final) ----------
    //     const std::complex<double> Em_f = E_fwd_local.m * amp;
    //     const std::complex<double> En_f = E_fwd_local.n * amp;

    //     // Proyección a lab (x,y)
    //     const std::complex<double> Efx = Em_f * Q(0, 0) + En_f * Q(1, 0) * -1.0;
    //     const std::complex<double> Efy = Em_f * Q(0, 1) + En_f * Q(1, 1);

    //     // -------- Reverse field (usa polarization_reverse) -------------
    //     const std::complex<double> Em_r = E_rev_local.m * amp * path_phase;
    //     const std::complex<double> En_r = E_rev_local.n * amp * path_phase;

    //     const std::complex<double> Erx = Em_r * Q(0, 0) + En_r * Q(1, 0) * -1.0;
    //     const std::complex<double> Ery = Em_r * Q(0, 1) + En_r * Q(1, 1);

    //     // Coherente: |E_f + E_r|^2
    //     const std::complex<double> Etx = Efx + Erx;
    //     const std::complex<double> Ety = Efy + Ery;

    //     const double S0c = std::norm(Etx) + std::norm(Ety);
    //     const double S1c = std::norm(Etx) - std::norm(Ety);
    //     const double S2c = 2.0 * std::real(Etx * std::conj(Ety));
    //     const double S3c = 2.0 * std::imag(Etx * std::conj(Ety));

    //     S0_coh(i, j) += S0c;
    //     S1_coh(i, j) += S1c;
    //     S2_coh(i, j) += S2c;
    //     S3_coh(i, j) += S3c;

    //     // Incoherente: |E_f|^2 + |E_r|^2
    //     const double S0i = (std::norm(Efx) + std::norm(Efy)) + (std::norm(Erx) + std::norm(Ery));
    //     const double S1i = (std::norm(Efx) - std::norm(Efy)) + (std::norm(Erx) - std::norm(Ery));
    //     const double S2i = 2.0 * (std::real(Efx * std::conj(Efy)) + std::real(Erx * std::conj(Ery)));
    //     const double S3i = 2.0 * (std::imag(Efx * std::conj(Efy)) + std::imag(Erx * std::conj(Ery)));

    //     S0_incoh(i, j) += S0i;
    //     S1_incoh(i, j) += S1i;
    //     S2_incoh(i, j) += S2i;
    //     S3_incoh(i, j) += S3i;
    //   }
    // }

    // hits++;
  }

  CVec2 coherent_estimation(const Photon &photon, const Medium &medium, Matrix last_scattering_P)
  {
    // --- Extrae bases y direcciones ---
    Vec3 s0 = row_vec3(photon.P0, 2);
    Vec3 s1 = row_vec3(photon.P1, 2);
    Vec3 snm1 = row_vec3(photon.Pn1, 2);
    Vec3 sn = row_vec3(last_scattering_P, 2);

    Vec3 m0 = row_vec3(photon.P0, 0);
    Vec3 n0 = row_vec3(photon.P0, 1);

    Vec3 m1 = row_vec3(photon.P1, 0);
    Vec3 n1 = row_vec3(photon.P1, 1);

    Vec3 mnm1 = row_vec3(photon.Pn1, 0);
    Vec3 nnm1 = row_vec3(photon.Pn1, 1);

    Vec3 mn = row_vec3(last_scattering_P, 0);
    Vec3 nn = row_vec3(last_scattering_P, 1);

    // --- Matriz Q ---
    CMatrix Q(2, 2);
    Q(0, 0) = 1;
    Q(0, 1) = 0;
    Q(1, 0) = 0;
    Q(1, 1) = -1;

    // --- T^T (transpose, NO conjugate) ---
    CMatrix Tt(2, 2);
    Tt(0, 0) = photon.matrix_T(0, 0);
    Tt(0, 1) = photon.matrix_T(1, 0);
    Tt(1, 0) = photon.matrix_T(0, 1);
    Tt(1, 1) = photon.matrix_T(1, 1);

    // =========================
    // Reverse: primer scattering (en r_n)
    // s_in = s0, s_out = -s_{n-1}
    // =========================
    Vec3 s_in_a = s0;
    Vec3 s_out_a = snm1 * (-1.0);

    double cos_th_a = clamp_pm1(dot(s_in_a, s_out_a));
    double th_a = std::acos(cos_th_a);
    CMatrix S_a = medium.scattering_matrix(th_a, 0.0, photon.k);

    // normal del plano n' = s_in x s_out (paper)
    Vec3 nprime = safe_unit(cross(s_in_a, s_out_a), n0);
    Vec3 mprime_in = safe_unit(cross(nprime, s_in_a), m0);
    Vec3 mprime_out = safe_unit(cross(nprime, s_out_a), mnm1);

    // R(phi_n): base (m0,n0) -> (mprime_in,nprime)
    CMatrix Rn = rot2(mprime_in, nprime, m0, n0);

    // scattering en el plano: salida en (mprime_out, nprime)
    CVec2 E = scatter_event(S_a, Rn, photon.initial_polarization);

    // R(phi_n'): (mprime_out,nprime) -> (m_{n-1}, -n_{n-1}) en dirección -s_{n-1}
    Vec3 n_to_mid_start = nnm1 * (-1.0);
    CMatrix Rnp = rot2(mnm1, n_to_mid_start, mprime_out, nprime);
    E = apply2(Rnp, E);

    // =========================
    // Bloque medio: Q T^T Q
    // =========================
    // E <- Q ( T^T ( Q E ) )
    E = apply2(Q, E);
    E = apply2(Tt, E);
    E = apply2(Q, E);

    // Ahora E está en base (m1, -n1) con dirección -s1

    // =========================
    // Reverse: último scattering (en r_1)
    // s_in = -s1, s_out = sn
    // =========================
    Vec3 s_in_b = s1 * (-1.0);
    Vec3 s_out_b = sn;

    double cos_th_b = clamp_pm1(dot(s_in_b, s_out_b));
    double th_b = std::acos(cos_th_b);
    CMatrix S_b = medium.scattering_matrix(th_b, 0.0, photon.k);

    // normal n'' = (-s1) x sn
    Vec3 npp = safe_unit(cross(s_in_b, s_out_b), n1 * (-1.0));
    Vec3 mpp_in = safe_unit(cross(npp, s_in_b), m1);
    Vec3 mpp_out = safe_unit(cross(npp, s_out_b), mn);

    // R(phi_1'): (m1,-n1) -> (mpp_in,npp)
    Vec3 n_mid_end = n1 * (-1.0);
    CMatrix R1p = rot2(mpp_in, npp, m1, n_mid_end);

    // scattering: salida en (mpp_out, npp)
    E = scatter_event(S_b, R1p, E);

    // Rotación final: (mpp_out,npp) -> (mn,nn) (tu base Pn)
    CMatrix Rout = rot2(mn, nn, mpp_out, npp);
    E = apply2(Rout, E);

    return E;
  }

  void coherent_calculation(Photon &photon, const Medium &medium)
  {
    // --- Extrae bases y direcciones ---
    Vec3 s0 = row_vec3(photon.P0, 2);
    Vec3 s1 = row_vec3(photon.P1, 2);
    Vec3 snm1 = row_vec3(photon.Pn1, 2);
    Vec3 sn = row_vec3(photon.Pn, 2);

    Vec3 m0 = row_vec3(photon.P0, 0);
    Vec3 n0 = row_vec3(photon.P0, 1);

    Vec3 m1 = row_vec3(photon.P1, 0);
    Vec3 n1 = row_vec3(photon.P1, 1);

    Vec3 mnm1 = row_vec3(photon.Pn1, 0);
    Vec3 nnm1 = row_vec3(photon.Pn1, 1);

    Vec3 mn = row_vec3(photon.Pn, 0);
    Vec3 nn = row_vec3(photon.Pn, 1);

    // --- Matriz Q ---
    CMatrix Q(2, 2);
    Q(0, 0) = 1;
    Q(0, 1) = 0;
    Q(1, 0) = 0;
    Q(1, 1) = -1;

    // --- T^T (transpose, NO conjugate) ---
    CMatrix Tt(2, 2);
    Tt(0, 0) = photon.matrix_T(0, 0);
    Tt(0, 1) = photon.matrix_T(1, 0);
    Tt(1, 0) = photon.matrix_T(0, 1);
    Tt(1, 1) = photon.matrix_T(1, 1);

    // =========================
    // Reverse: primer scattering (en r_n)
    // s_in = s0, s_out = -s_{n-1}
    // =========================
    Vec3 s_in_a = s0;
    Vec3 s_out_a = snm1 * (-1.0);

    double cos_th_a = clamp_pm1(dot(s_in_a, s_out_a));
    double th_a = std::acos(cos_th_a);
    CMatrix S_a = medium.scattering_matrix(th_a, 0.0, photon.k);

    // normal del plano n' = s_in x s_out (paper)
    Vec3 nprime = safe_unit(cross(s_in_a, s_out_a), n0);
    Vec3 mprime_in = safe_unit(cross(nprime, s_in_a), m0);
    Vec3 mprime_out = safe_unit(cross(nprime, s_out_a), mnm1);

    // R(phi_n): base (m0,n0) -> (mprime_in,nprime)
    CMatrix Rn = rot2(mprime_in, nprime, m0, n0);

    // scattering en el plano: salida en (mprime_out, nprime)
    CVec2 E = scatter_event(S_a, Rn, photon.initial_polarization);

    // R(phi_n'): (mprime_out,nprime) -> (m_{n-1}, -n_{n-1}) en dirección -s_{n-1}
    Vec3 n_to_mid_start = nnm1 * (-1.0);
    CMatrix Rnp = rot2(mnm1, n_to_mid_start, mprime_out, nprime);
    E = apply2(Rnp, E);

    // =========================
    // Bloque medio: Q T^T Q
    // =========================
    // E <- Q ( T^T ( Q E ) )
    E = apply2(Q, E);
    E = apply2(Tt, E);
    E = apply2(Q, E);

    // Ahora E está en base (m1, -n1) con dirección -s1

    // =========================
    // Reverse: último scattering (en r_1)
    // s_in = -s1, s_out = sn
    // =========================
    Vec3 s_in_b = s1 * (-1.0);
    Vec3 s_out_b = sn;

    double cos_th_b = clamp_pm1(dot(s_in_b, s_out_b));
    double th_b = std::acos(cos_th_b);
    CMatrix S_b = medium.scattering_matrix(th_b, 0.0, photon.k);

    // normal n'' = (-s1) x sn
    Vec3 npp = safe_unit(cross(s_in_b, s_out_b), n1 * (-1.0));
    Vec3 mpp_in = safe_unit(cross(npp, s_in_b), m1);
    Vec3 mpp_out = safe_unit(cross(npp, s_out_b), mn);

    // R(phi_1'): (m1,-n1) -> (mpp_in,npp)
    Vec3 n_mid_end = n1 * (-1.0);
    CMatrix R1p = rot2(mpp_in, npp, m1, n_mid_end);

    // scattering: salida en (mpp_out, npp)
    E = scatter_event(S_b, R1p, E);

    // Rotación final: (mpp_out,npp) -> (mn,nn) (tu base Pn)
    CMatrix Rout = rot2(mn, nn, mpp_out, npp);
    E = apply2(Rout, E);

    photon.polarization_reverse = E;
  }

  // StatisticsSensor implementation
  StatisticsSensor::StatisticsSensor(double z) : Sensor(z)
  {
  }

  std::unique_ptr<Sensor> StatisticsSensor::clone() const
  {
    auto det = std::make_unique<StatisticsSensor>(origin.z);
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
