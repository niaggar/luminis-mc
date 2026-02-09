#include <cmath>
#include <complex>
#include <luminis/core/detector.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/core/medium.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/log/logger.hpp>
#include <vector>
#include <functional>

namespace luminis::core
{
  // MultiDetector implementation
  void MultiDetector::add_detector(std::unique_ptr<Detector> detector)
  {
    u_int new_id = static_cast<u_int>(detectors.size());
    detector->id = new_id;
    detectors.push_back(std::move(detector));
  }

  bool MultiDetector::record_hit(Photon &photon, std::function<void()> coherent_calculation)
  {
    bool hit_recorded = false;
    for (const auto &detector : detectors)
    {
      if (detector->record_hit(photon, coherent_calculation))
      {
        hit_recorded = true;
      }
    }
    return hit_recorded;
  }

  void MultiDetector::merge_from(const MultiDetector &other)
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
        LLOG_WARN("MultiDetector merge_from: Detector ID {} not found in destination", det_id);
      }
    }
  }

  std::unique_ptr<MultiDetector> MultiDetector::clone() const
  {
    auto multi_det = std::make_unique<MultiDetector>();
    for (const auto &detector : detectors)
    {
      multi_det->add_detector(detector->clone());
    }
    return multi_det;
  }

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

  void Detector::merge_from(const Detector &other)
  {
    hits += other.hits;
    recorded_photons.insert(recorded_photons.end(),
                            other.recorded_photons.begin(),
                            other.recorded_photons.end());
  }

  bool Detector::record_hit(Photon &photon, std::function<void()> coherent_calculation)
  {
    const bool valid = check_conditions(photon);
    if (!valid)
      return false;

    const Vec3 xn = photon.prev_pos;
    const Vec3 xf = photon.pos;
    const Vec3 d = xf - xn;

    const double denom = dot(d, normal);
    if (std::abs(denom) < 1e-6)
      return false;

    const double t = dot(origin - xn, normal) / denom;
    if (t < 0 || t > 1)
      return false;

    const Vec3 hit_point = xn + d * t;
    const double correction_distance = luminis::math::norm(hit_point - xf);
    if (correction_distance > 0)
    {
      photon.opticalpath -= correction_distance;
    }
    hits += 1;
    photon.detected_pos = hit_point;

    PhotonRecord photon_rec{};
    photon_rec.events = photon.events;
    photon_rec.penetration_depth = photon.penetration_depth;
    photon_rec.launch_time = photon.launch_time;
    photon_rec.arrival_time = photon.launch_time + (photon.opticalpath / photon.velocity);
    photon_rec.opticalpath = photon.opticalpath;
    photon_rec.weight = photon.weight;
    photon_rec.k = photon.k;
    photon_rec.position_detector = photon.detected_pos;
    photon_rec.position_first_scattering = photon.r_0;
    photon_rec.position_last_scattering = photon.r_n;
    photon_rec.direction = photon.dir;
    photon_rec.m = photon.m;
    photon_rec.n = photon.n;
    photon_rec.polarization_forward = photon.polarization;
    photon_rec.polarization_reverse = CVec2{std::complex<double>(0, 0), std::complex<double>(0, 0)};

    recorded_photons.push_back(photon_rec);
    return true;
  }

  void Detector::set_theta_limit(double min, double max)
  {
    filter_theta_enabled = true;
    filter_theta_min = min;
    filter_theta_max = max;
    double c1 = std::cos(min);
    double c2 = std::cos(max);
    _cache_cos_theta_min = std::min(c1, c2);
    _cache_cos_theta_max = std::max(c1, c2);
  }

  void Detector::set_phi_limit(double min, double max)
  {
    filter_phi_enabled = true;
    filter_phi_min = min;
    filter_phi_max = max;
  }

  bool Detector::check_conditions(const Photon &photon) const
  {
    if (filter_theta_enabled)
    {
      const double costtheta = -1 * photon.dir.z;
      if (costtheta < _cache_cos_theta_min || costtheta > _cache_cos_theta_max)
        return false;
    }

    if (filter_phi_enabled)
    {
      double phi = std::atan2(photon.dir.y, photon.dir.x);
      if (phi < 0)
        phi += 2.0 * M_PI;
      if (phi < filter_phi_min || phi > filter_phi_max)
        return false;
    }

    return true;
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

  bool AngleDetector::record_hit(Photon &photon, std::function<void()> coherent_calculation)
  {
    if (photon.pos.z > 0)
      return false;

    const bool valid = check_conditions(photon);
    if (!valid)
      return false;

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
    return true;
  }

  std::unique_ptr<Detector> AngleDetector::clone() const
  {
    auto det = std::make_unique<AngleDetector>(origin.z, N_theta, N_phi);
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

  // HistogramDetector implementation
  bool HistogramDetector::record_hit(Photon &photon, std::function<void()> coherent_calculation)
  {
    if (photon.pos.z > 0)
      return false;

    const bool valid = check_conditions(photon);
    if (!valid)
      return false;

    int event = static_cast<int>(photon.events);
    if (event >= 0 && event < static_cast<int>(histogram.size()))
    {
      histogram[event] += 1;
      hits += 1;
    }

    return true;
  }

  std::unique_ptr<Detector> HistogramDetector::clone() const
  {
    auto det = std::make_unique<HistogramDetector>(origin.z, histogram.size());
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

  void HistogramDetector::merge_from(const Detector &other)
  {
    const HistogramDetector &other_hist = dynamic_cast<const HistogramDetector &>(other);
    hits += other_hist.hits;
    for (size_t i = 0; i < histogram.size(); ++i)
    {
      histogram[i] += other_hist.histogram[i];
    }
  }

  // ThetaHistogramDetector implementation
  bool ThetaHistogramDetector::record_hit(Photon &photon, std::function<void()> coherent_calculation)
  {
    if (photon.pos.z > 0)
      return false;

    const bool valid = check_conditions(photon);
    if (!valid)
      return false;

    const Vec3 u = photon.dir;
    const double costtheta = -1 * u.z;
    const double theta = std::acos(costtheta);

    const int itheta = std::min(static_cast<int>(std::floor((theta / (M_PI / 2.0)) * histogram.size())), static_cast<int>(histogram.size() - 1));
    histogram[itheta] += 1;
    hits += 1;

    return true;
  }

  std::unique_ptr<Detector> ThetaHistogramDetector::clone() const
  {
    auto det = std::make_unique<ThetaHistogramDetector>(origin.z, histogram.size());
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

  void ThetaHistogramDetector::merge_from(const Detector &other)
  {
    const ThetaHistogramDetector &other_hist = dynamic_cast<const ThetaHistogramDetector &>(other);
    hits += other_hist.hits;
    for (size_t i = 0; i < histogram.size(); ++i)
    {
      histogram[i] += other_hist.histogram[i];
    }
  }

  // SpatialDetector implementation
  SpatialDetector::SpatialDetector(double z, double x_len, double y_len, double r_len, int n_x, int n_y, int n_r)
      : Detector(z)
  {
    N_x = n_x;
    N_y = n_y;
    N_r = n_r;
    dr = r_len / N_r;
    dx = x_len / n_x;
    dy = y_len / n_y;
    E_x = CMatrix(n_x, n_y);
    E_y = CMatrix(n_x, n_y);
    E_z = CMatrix(n_x, n_y);
    I_x = Matrix(n_x, n_y);
    I_y = Matrix(n_x, n_y);
    I_z = Matrix(n_x, n_y);
    I_plus = Matrix(n_x, n_y);
    I_minus = Matrix(n_x, n_y);
    I_rad_plus.resize(N_r, 0.0);
    I_rad_minus.resize(N_r, 0.0);
  }

  bool SpatialDetector::record_hit(Photon &photon, std::function<void()> coherent_calculation)
  {
    const bool valid = check_conditions(photon);
    if (!valid)
      return false;

    const Vec3 xn = photon.prev_pos;
    const Vec3 xf = photon.pos;
    const Vec3 d = xf - xn;

    const double denom = dot(d, normal);
    if (std::abs(denom) < 1e-6)
      return false;

    const double t = dot(origin - xn, normal) / denom;
    if (t < 0 || t > 1)
      return false;

    const Vec3 hit_point = xn + d * t;
    const double correction_distance = luminis::math::norm(hit_point - xf);
    if (correction_distance > 0)
    {
      photon.opticalpath -= correction_distance;
    }

    const double length_x = N_x * dx;
    const double length_y = N_y * dy;
    const double min_x = -0.5 * length_x;
    const double min_y = -0.5 * length_y;
    const double max_x = 0.5 * length_x;
    const double max_y = 0.5 * length_y;

    // Validate position within detector area
    if (hit_point.x < min_x || hit_point.x >= max_x || hit_point.y < min_y || hit_point.y >= max_y)
    {
      return false;
    }

    // Determine bins
    const int ix = std::min(static_cast<int>(std::floor(((hit_point.x - min_x) / length_x) * N_x)), N_x - 1);
    const int iy = std::min(static_cast<int>(std::floor(((hit_point.y - min_y) / length_y) * N_y)), N_y - 1);

    if (ix < 0 || ix >= N_x || iy < 0 || iy >= N_y)
      return false;

    hits += 1;
    photon.detected_pos = hit_point;

    const double w = photon.weight;

    // Compute local field contribution
    std::complex<double> phase = std::exp(std::complex<double>(0, photon.k * photon.opticalpath));
    std::complex<double> Em_local_photon = photon.polarization.m * phase * std::sqrt(photon.weight);
    std::complex<double> En_local_photon = photon.polarization.n * phase * std::sqrt(photon.weight);

    std::complex<double> E_det_x = (Em_local_photon * photon.m.x + En_local_photon * photon.n.x);
    std::complex<double> E_det_y = (Em_local_photon * photon.m.y + En_local_photon * photon.n.y);
    std::complex<double> E_det_z = (Em_local_photon * photon.m.z + En_local_photon * photon.n.z);

    const std::complex<double> I_imag(0.0, 1.0);
    std::complex<double> E_plus_val = (E_det_x - I_imag * E_det_y) / std::sqrt(2.0);
    std::complex<double> E_minus_val = (E_det_x + I_imag * E_det_y) / std::sqrt(2.0);

    // Accumulate field contributions
    E_x(ix, iy) += Em_local_photon * photon.m.x + En_local_photon * photon.n.x;
    E_y(ix, iy) += Em_local_photon * photon.m.y + En_local_photon * photon.n.y;
    E_z(ix, iy) += Em_local_photon * photon.m.z + En_local_photon * photon.n.z;

    I_x(ix, iy) += std::norm(E_det_x);
    I_y(ix, iy) += std::norm(E_det_y);
    I_z(ix, iy) += std::norm(E_det_z);
    I_plus(ix, iy) += std::norm(E_plus_val);
    I_minus(ix, iy) += std::norm(E_minus_val);

    // Radial bins
    double r = std::sqrt(hit_point.x * hit_point.x + hit_point.y * hit_point.y);
    int ir = static_cast<int>(r / dr);
    if (ir < N_r)
    {
      I_rad_plus[ir] += std::norm(E_plus_val);
      I_rad_minus[ir] += std::norm(E_minus_val);
    }

    return true;
  }

  std::unique_ptr<Detector> SpatialDetector::clone() const
  {
    const double x_len = N_x * dx;
    const double y_len = N_y * dy;
    const double r_len = N_r * dr;
    auto det = std::make_unique<SpatialDetector>(origin.z, x_len, y_len, r_len, N_x, N_y, N_r);
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

  void SpatialDetector::merge_from(const Detector &other)
  {
    const SpatialDetector &other_spatial = dynamic_cast<const SpatialDetector &>(other);
    hits += other_spatial.hits;

    for (int ix = 0; ix < N_x; ++ix)
    {
      for (int iy = 0; iy < N_y; ++iy)
      {
        E_x(ix, iy) += other_spatial.E_x(ix, iy);
        E_y(ix, iy) += other_spatial.E_y(ix, iy);
        E_z(ix, iy) += other_spatial.E_z(ix, iy);
        I_x(ix, iy) += other_spatial.I_x(ix, iy);
        I_y(ix, iy) += other_spatial.I_y(ix, iy);
        I_z(ix, iy) += other_spatial.I_z(ix, iy);
        I_plus(ix, iy) += other_spatial.I_plus(ix, iy);
        I_minus(ix, iy) += other_spatial.I_minus(ix, iy);
      }
    }

    for (int ir = 0; ir < N_r; ++ir)
    {
      I_rad_plus[ir] += other_spatial.I_rad_plus[ir];
      I_rad_minus[ir] += other_spatial.I_rad_minus[ir];
    }
  }

  std::vector<double> SpatialDetector::calculate_radial_plus_intensity() const
  {
    std::vector<double> radial_intensity(N_r, 0.0);
    for (int ir = 0; ir < N_r; ++ir)
    {
      double r_inner = ir * dr;
      double r_outer = (ir + 1) * dr;
      double area = M_PI * (r_outer * r_outer - r_inner * r_inner);
      if (area > 0)
      {
        radial_intensity[ir] = I_rad_plus[ir] / area;
      }
    }
    return radial_intensity;
  }

  std::vector<double> SpatialDetector::calculate_radial_minus_intensity() const
  {
    std::vector<double> radial_intensity(N_r, 0.0);
    for (int ir = 0; ir < N_r; ++ir)
    {
      double r_inner = ir * dr;
      double r_outer = (ir + 1) * dr;
      double area = M_PI * (r_outer * r_outer - r_inner * r_inner);
      if (area > 0)
      {
        radial_intensity[ir] = I_rad_minus[ir] / area;
      }
    }
    return radial_intensity;
  }

  // SpatialTimeDetector implementation
  SpatialTimeDetector::SpatialTimeDetector(double z, double x_len, double y_len, double r_len,int n_x, int n_y, int n_r, int n_t, double dt, double t_max)
      : Detector(z)
  {
    N_x = n_x;
    N_y = n_y;
    N_r = n_r;
    N_t = n_t;
    dx = x_len / n_x;
    dy = y_len / n_y;
    dr = r_len / n_r;
    this->dt = dt;
    this->t_max = t_max;
    time_bins.reserve(N_t);
    for (int i = 0; i < N_t; ++i)
    {
      time_bins.push_back(SpatialDetector(z, x_len, y_len, r_len, n_x, n_y, n_r));
    }
  }

  bool SpatialTimeDetector::record_hit(Photon &photon, std::function<void()> coherent_calculation)
  {
    const bool valid = check_conditions(photon);
    if (!valid)
      return false;

    const double arrival_time = photon.launch_time + (photon.opticalpath / photon.velocity);
    if (arrival_time < 0 || arrival_time >= t_max)
      return false;

    int time_bin_index = static_cast<int>(arrival_time / dt);
    if (time_bin_index < 0 || time_bin_index >= N_t)
      return false;

    return time_bins[time_bin_index].record_hit(photon, coherent_calculation);
  }

  std::unique_ptr<Detector> SpatialTimeDetector::clone() const
  {
    const double x_len = N_x * dx;
    const double y_len = N_y * dy;
    const double r_len = N_r * dr;
    auto det = std::make_unique<SpatialTimeDetector>(origin.z, x_len, y_len, r_len, N_x, N_y, N_r, N_t, dt, t_max);
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

  void SpatialTimeDetector::merge_from(const Detector &other)
  {
    const SpatialTimeDetector &other_time = dynamic_cast<const SpatialTimeDetector &>(other);
    hits += other_time.hits;

    for (int i = 0; i < N_t; ++i)
    {
      time_bins[i].merge_from(other_time.time_bins[i]);
    }
  }

  // SpatialCoherentDetector implementation
  SpatialCoherentDetector::SpatialCoherentDetector(double z, double x_len, double y_len, int n_x, int n_y)
      : Detector(z)
  {
    N_x = n_x;
    N_y = n_y;
    dx = x_len / n_x;
    dy = y_len / n_y;
    I_x = Matrix(N_x, N_y);
    I_y = Matrix(N_x, N_y);
    I_z = Matrix(N_x, N_y);
    I_inco_x = Matrix(N_x, N_y);
    I_inco_y = Matrix(N_x, N_y);
    I_inco_z = Matrix(N_x, N_y);

    I_x_theta.resize(200, 0.0);
    I_y_theta.resize(200, 0.0);
    I_z_theta.resize(200, 0.0);

    I_inco_x_theta.resize(200, 0.0);
    I_inco_y_theta.resize(200, 0.0);
    I_inco_z_theta.resize(200, 0.0);
  }

  bool SpatialCoherentDetector::record_hit(Photon &photon, std::function<void()> coherent_calculation)
  {
    if (photon.events < 2)
      return false;

    const bool valid = check_conditions(photon);
    if (!valid)
      return false;

    const Vec3 xn = photon.prev_pos;
    const Vec3 xf = photon.pos;
    const Vec3 d = xf - xn;

    const double denom = dot(d, normal);
    if (std::abs(denom) < 1e-6)
      return false;

    const double t = dot(origin - xn, normal) / denom;
    if (t < 0 || t > 1)
      return false;

    const Vec3 hit_point = xn + d * t;
    const double correction_distance = luminis::math::norm(hit_point - xf);
    if (correction_distance > 0)
    {
      photon.opticalpath -= correction_distance;
    }
    hits += 1;
    photon.detected_pos = hit_point;

    const double length_x = N_x * dx;
    const double length_y = N_y * dy;
    const double min_x = -0.5 * length_x;
    const double min_y = -0.5 * length_y;
    const double max_x = 0.5 * length_x;
    const double max_y = 0.5 * length_y;

    // Validate position within detector area
    if (photon.detected_pos.x < min_x || photon.detected_pos.x >= max_x || photon.detected_pos.y < min_y || photon.detected_pos.y >= max_y)
    {
      return false;
    }

    // Determine bins
    const int ix = std::min(static_cast<int>(std::floor(((photon.detected_pos.x - min_x) / length_x) * N_x)), N_x - 1);
    const int iy = std::min(static_cast<int>(std::floor(((photon.detected_pos.y - min_y) / length_y) * N_y)), N_y - 1);

    if (ix < 0 || ix >= N_x || iy < 0 || iy >= N_y)
      return false;

    coherent_calculation();

    // Compute intensity contribution
    double w = photon.weight;

    Vec3 qb = (photon.dir + photon.s_0) * photon.k;
    Vec3 delta_r = photon.r_n - photon.r_0;
    std::complex<double> path_phase = std::exp(std::complex<double>(0, dot(qb, delta_r)));
    std::complex<double> phase = std::exp(std::complex<double>(0, photon.k * photon.opticalpath));

    Vec3 n_0 = photon.n_0;
    Vec3 s_0 = photon.s_0;
    Vec3 m_0 = cross(n_0, s_0);

    CVec2 E_fwd_local = photon.polarization;
    CVec2 E_rev_local = photon.polarization_reverse;

    std::complex<double> E_fwd_lab_x = (E_fwd_local.m * photon.m.x + E_fwd_local.n * photon.n.x) * phase * std::sqrt(w);
    std::complex<double> E_fwd_lab_y = (E_fwd_local.m * photon.m.y + E_fwd_local.n * photon.n.y) * phase * std::sqrt(w);
    std::complex<double> E_fwd_lab_z = (E_fwd_local.m * photon.m.z + E_fwd_local.n * photon.n.z) * phase * std::sqrt(w);

    std::complex<double> E_rev_lab_x = (E_rev_local.m * m_0.x + E_rev_local.n * n_0.x) * phase * path_phase * std::sqrt(w);
    std::complex<double> E_rev_lab_y = (E_rev_local.m * m_0.y + E_rev_local.n * n_0.y) * phase * path_phase * std::sqrt(w);
    std::complex<double> E_rev_lab_z = (E_rev_local.m * m_0.z + E_rev_local.n * n_0.z) * phase * path_phase * std::sqrt(w);

    // Accumulate coherent intensity contributions
    I_x(ix, iy) += std::norm(E_fwd_lab_x + E_rev_lab_x);
    I_y(ix, iy) += std::norm(E_fwd_lab_y + E_rev_lab_y);
    I_z(ix, iy) += std::norm(E_fwd_lab_z + E_rev_lab_z);

    // Accumulate incoherent intensity contributions
    I_inco_x(ix, iy) += std::norm(E_fwd_lab_x) + std::norm(E_rev_lab_x);
    I_inco_y(ix, iy) += std::norm(E_fwd_lab_y) + std::norm(E_rev_lab_y);
    I_inco_z(ix, iy) += std::norm(E_fwd_lab_z) + std::norm(E_rev_lab_z);
    return true;
  }

  std::unique_ptr<Detector> SpatialCoherentDetector::clone() const
  {
    auto det = std::make_unique<SpatialCoherentDetector>(origin.z, N_x * dx, N_y * dy, N_x, N_y);
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

  void SpatialCoherentDetector::merge_from(const Detector &other)
  {
    const SpatialCoherentDetector &other_spatial = dynamic_cast<const SpatialCoherentDetector &>(other);
    hits += other_spatial.hits;

    for (int ix = 0; ix < N_x; ++ix)
    {
      for (int iy = 0; iy < N_y; ++iy)
      {
        I_x(ix, iy) += other_spatial.I_x(ix, iy);
        I_y(ix, iy) += other_spatial.I_y(ix, iy);
        I_z(ix, iy) += other_spatial.I_z(ix, iy);
        I_inco_x(ix, iy) += other_spatial.I_inco_x(ix, iy);
        I_inco_y(ix, iy) += other_spatial.I_inco_y(ix, iy);
        I_inco_z(ix, iy) += other_spatial.I_inco_z(ix, iy);
      }
    }
  }

  // AngularCoherentDetector implementation
  AngularCoherentDetector::AngularCoherentDetector(double z, int n_theta, double max_theta)
      : Detector(z)
  {
    N_theta = n_theta;
    dtheta = max_theta / n_theta;
    I_x = std::vector<double>(N_theta, 0.0);
    I_y = std::vector<double>(N_theta, 0.0);
    I_z = std::vector<double>(N_theta, 0.0);
    I_plus = std::vector<double>(N_theta, 0.0);
    I_minus = std::vector<double>(N_theta, 0.0);
    I_total = std::vector<double>(N_theta, 0.0);

    I_inco_x = std::vector<double>(N_theta, 0.0);
    I_inco_y = std::vector<double>(N_theta, 0.0);
    I_inco_z = std::vector<double>(N_theta, 0.0);
    I_inco_plus = std::vector<double>(N_theta, 0.0);
    I_inco_minus = std::vector<double>(N_theta, 0.0);
    I_inco_total = std::vector<double>(N_theta, 0.0);

    set_theta_limit(0.0, max_theta);
  }

  bool AngularCoherentDetector::record_hit(Photon &photon, std::function<void()> coherent_calculation)
  {
    if (photon.pos.z > 0)
      return false;
    if (photon.events < 2)
      return false;

    const bool valid = check_conditions(photon);
    if (!valid)
      return false;

    const Vec3 u = photon.dir;
    const double costtheta = -1 * u.z;
    const double theta = std::acos(costtheta);

    const int itheta = std::min(static_cast<int>(std::floor(N_theta * (theta / filter_theta_max))), N_theta - 1);

    coherent_calculation();

    // Compute intensity contribution
    double w = photon.weight;

    Vec3 qb = (photon.s_n + photon.s_0) * photon.k;
    Vec3 delta_r = photon.r_n - photon.r_0;
    std::complex<double> path_phase = std::exp(std::complex<double>(0, dot(qb, delta_r)));
    std::complex<double> phase = std::exp(std::complex<double>(0, photon.k * photon.opticalpath));

    Vec3 n_0 = photon.n_0;
    Vec3 s_0 = photon.s_0;
    Vec3 m_0 = cross(n_0, s_0);

    CVec2 E_fwd_local = photon.polarization;
    CVec2 E_rev_local = photon.polarization_reverse;

    std::complex<double> E_fwd_lab_x = (E_fwd_local.m * photon.m.x + E_fwd_local.n * photon.n.x) * phase * std::sqrt(w);
    std::complex<double> E_fwd_lab_y = (E_fwd_local.m * photon.m.y + E_fwd_local.n * photon.n.y) * phase * std::sqrt(w);
    std::complex<double> E_fwd_lab_z = (E_fwd_local.m * photon.m.z + E_fwd_local.n * photon.n.z) * phase * std::sqrt(w);

    std::complex<double> E_rev_lab_x = (E_rev_local.m * m_0.x + E_rev_local.n * n_0.x) * phase * path_phase * std::sqrt(w);
    std::complex<double> E_rev_lab_y = (E_rev_local.m * m_0.y + E_rev_local.n * n_0.y) * phase * path_phase * std::sqrt(w);
    std::complex<double> E_rev_lab_z = (E_rev_local.m * m_0.z + E_rev_local.n * n_0.z) * phase * path_phase * std::sqrt(w);

    std::complex<double> Etot_x = E_fwd_lab_x + E_rev_lab_x;
    std::complex<double> Etot_y = E_fwd_lab_y + E_rev_lab_y;

    const std::complex<double> I(0.0, 1.0);
    std::complex<double> E_plus_val = (Etot_x - I * Etot_y) / std::sqrt(2.0);
    std::complex<double> E_minus_val = (Etot_x + I * Etot_y) / std::sqrt(2.0);

    // Accumulate coherent intensity contributions
    I_x[itheta] += std::norm(E_fwd_lab_x + E_rev_lab_x);
    I_y[itheta] += std::norm(E_fwd_lab_y + E_rev_lab_y);
    I_z[itheta] += std::norm(E_fwd_lab_z + E_rev_lab_z);
    I_plus[itheta] += std::norm(E_plus_val);
    I_minus[itheta] += std::norm(E_minus_val);
    I_total[itheta] += std::norm(E_fwd_lab_x + E_rev_lab_x + E_fwd_lab_y + E_rev_lab_y);

    std::complex<double> E_fwd_plus = (E_fwd_lab_x - I * E_fwd_lab_y) / std::sqrt(2.0);
    std::complex<double> E_fwd_minus = (E_fwd_lab_x + I * E_fwd_lab_y) / std::sqrt(2.0);
    std::complex<double> E_rev_plus = (E_rev_lab_x - I * E_rev_lab_y) / std::sqrt(2.0);
    std::complex<double> E_rev_minus = (E_rev_lab_x + I * E_rev_lab_y) / std::sqrt(2.0);

    // Accumulate incoherent intensity contributions
    I_inco_x[itheta] += std::norm(E_fwd_lab_x) + std::norm(E_rev_lab_x);
    I_inco_y[itheta] += std::norm(E_fwd_lab_y) + std::norm(E_rev_lab_y);
    I_inco_z[itheta] += std::norm(E_fwd_lab_z) + std::norm(E_rev_lab_z);
    I_inco_plus[itheta] += std::norm(E_fwd_plus) + std::norm(E_rev_plus);
    I_inco_minus[itheta] += std::norm(E_fwd_minus) + std::norm(E_rev_minus);
    I_inco_total[itheta] += std::norm(E_fwd_lab_x) + std::norm(E_rev_lab_x) + std::norm(E_fwd_lab_y) + std::norm(E_rev_lab_y);

    hits += 1;
    return true;
  }

  std::unique_ptr<Detector> AngularCoherentDetector::clone() const
  {
    auto det = std::make_unique<AngularCoherentDetector>(origin.z, N_theta, filter_theta_max);
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

  void AngularCoherentDetector::merge_from(const Detector &other)
  {
    const AngularCoherentDetector &other_angular = dynamic_cast<const AngularCoherentDetector &>(other);
    hits += other_angular.hits;

    for (int itheta = 0; itheta < N_theta; ++itheta)
    {
      I_x[itheta] += other_angular.I_x[itheta];
      I_y[itheta] += other_angular.I_y[itheta];
      I_z[itheta] += other_angular.I_z[itheta];
      I_plus[itheta] += other_angular.I_plus[itheta];
      I_minus[itheta] += other_angular.I_minus[itheta];
      I_total[itheta] += other_angular.I_total[itheta];

      I_inco_x[itheta] += other_angular.I_inco_x[itheta];
      I_inco_y[itheta] += other_angular.I_inco_y[itheta];
      I_inco_z[itheta] += other_angular.I_inco_z[itheta];
      I_inco_plus[itheta] += other_angular.I_inco_plus[itheta];
      I_inco_minus[itheta] += other_angular.I_inco_minus[itheta];
      I_inco_total[itheta] += other_angular.I_inco_total[itheta];
    }
  }

  // Detection condition factories
  DetectionCondition make_theta_condition(double min_theta, double max_theta)
  {
    return [min_theta, max_theta](const Photon &photon)
    {
      const Vec3 u = photon.dir;
      const double costtheta = -1 * u.z;
      const double theta = std::acos(costtheta);
      return (theta >= min_theta && theta <= max_theta);
    };
  };

  DetectionCondition make_phi_condition(double min_phi, double max_phi)
  {
    return [min_phi, max_phi](const Photon &photon)
    {
      const Vec3 u = photon.dir;
      double phi = std::atan2(u.y, u.x);
      if (phi < 0)
        phi += 2.0 * M_PI;
      return (phi >= min_phi && phi <= max_phi);
    };
  };

  DetectionCondition make_position_condition(double min_x, double max_x, double min_y, double max_y)
  {
    return [min_x, max_x, min_y, max_y](const Photon &photon)
    {
      const Vec3 pos = photon.pos;
      return (pos.x >= min_x && pos.x <= max_x && pos.y >= min_y && pos.y <= max_y);
    };
  };

  DetectionCondition make_events_condition(uint min_events, uint max_events)
  {
    return [min_events, max_events](const Photon &photon)
    {
      return (photon.events >= min_events && photon.events <= max_events);
    };
  };
} // namespace luminis::core
