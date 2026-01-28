#include <cmath>
#include <complex>
#include <luminis/core/detector.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/log/logger.hpp>
#include <vector>

namespace luminis::core
{
  // MultiDetector implementation
  void MultiDetector::add_detector(std::unique_ptr<Detector> detector)
  {
    u_int new_id = static_cast<u_int>(detectors.size());
    detector->id = new_id;
    detectors.push_back(std::move(detector));
  }

  std::vector<u_int> MultiDetector::validate_hit_by(const Photon &photon) const
  {
    std::vector<u_int> hit_detectors;
    for (const auto &detector : detectors)
    {
      if (detector->is_hit_by(photon))
      {
        hit_detectors.push_back(detector->id);
      }
    }
    return hit_detectors;
  }

  void MultiDetector::record_hit_by(Photon &photon)
  {
    for (const auto &detector : detectors)
    {
      detector->record_hit(photon);
    }
  }

  void MultiDetector::record_hit_in(Photon &photon, const std::vector<u_int> &detector_ids)
  {
    for (const auto &det_id : detector_ids)
    {
      if (det_id < detectors.size())
      {
        detectors[det_id]->record_hit(photon);
      }
      else
      {
        LLOG_WARN("MultiDetector record_hit_in: Detector ID {} not found", det_id);
      }
    }
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

  bool Detector::is_hit_by(const Photon &photon) const
  {
    const bool valid = check_conditions(photon);
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
    const bool valid = check_conditions(photon);
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

  void AngleDetector::record_hit(Photon &photon)
  {
    const bool valid = check_conditions(photon);
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
  void HistogramDetector::record_hit(Photon &photon)
  {
    const bool valid = check_conditions(photon);
    if (!valid)
      return;

    int event = static_cast<int>(photon.events);
    if (event >= 0 && event < static_cast<int>(histogram.size()))
    {
      histogram[event] += 1;
      hits += 1;
    }
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
  void ThetaHistogramDetector::record_hit(Photon &photon)
  {
    const bool valid = check_conditions(photon);
    if (!valid)
      return;

    const Vec3 u = photon.dir;
    const double costtheta = -1 * u.z;
    const double theta = std::acos(costtheta);

    const int itheta = std::min(static_cast<int>(std::floor((theta / (M_PI / 2.0)) * histogram.size())), static_cast<int>(histogram.size() - 1));
    histogram[itheta] += 1;
    hits += 1;
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
  SpatialDetector::SpatialDetector(double z, double x_len, double y_len, int n_x, int n_y)
      : Detector(z),
        N_x(n_x),
        N_y(n_y),
        dx(x_len / n_x),
        dy(y_len / n_y),
        E_x(n_x, n_y),
        E_y(n_x, n_y),
        E_z(n_x, n_y)
  {
  }

  void SpatialDetector::record_hit(Photon &photon)
  {
    const bool valid = check_conditions(photon);
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

    // Determine bins
    const int ix = static_cast<int>(std::floor((hit_point.x + (N_x * dx) / 2.0) / dx));
    const int iy = static_cast<int>(std::floor((hit_point.y + (N_y * dy) / 2.0) / dy));
    if (ix < 0 || ix >= N_x || iy < 0 || iy >= N_y)
      return;

    // Compute local field contribution
    std::complex<double> phase = std::exp(std::complex<double>(0, photon.k * photon.opticalpath));
    std::complex<double> Em_local_photon = photon.polarization.m * phase * std::sqrt(photon.weight);
    std::complex<double> En_local_photon = photon.polarization.n * phase * std::sqrt(photon.weight);

    // Accumulate field contributions
    E_x(ix, iy) += Em_local_photon * photon.m.x + En_local_photon * photon.n.x;
    E_y(ix, iy) += Em_local_photon * photon.m.y + En_local_photon * photon.n.y;
    E_z(ix, iy) += Em_local_photon * photon.m.z + En_local_photon * photon.n.z;
  }

  std::unique_ptr<Detector> SpatialDetector::clone() const
  {
    auto det = std::make_unique<SpatialDetector>(origin.z, N_x, N_y, dx, dy);
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
      }
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

    I_x_theta.resize(N_x, 0.0);
    I_y_theta.resize(N_x, 0.0);
    I_z_theta.resize(N_x, 0.0);

    I_inco_x_theta.resize(N_x, 0.0);
    I_inco_y_theta.resize(N_x, 0.0);
    I_inco_z_theta.resize(N_x, 0.0);
  }

  void SpatialCoherentDetector::record_hit(Photon &photon)
  {
    // LLOG_INFO("SpatialCoherentDetector: Recording hit");

    const bool valid = check_conditions(photon);
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

    const double length_x = N_x * dx;
    const double length_y = N_y * dy;
    const double min_x = -0.5 * length_x;
    const double min_y = -0.5 * length_y;
    const double max_x = 0.5 * length_x;
    const double max_y = 0.5 * length_y;

    // Validate position within detector area
    if (photon.pos.x < min_x || photon.pos.x >= max_x || photon.pos.y < min_y || photon.pos.y >= max_y)
    {
      return;
    }

    // Determine bins
    const int ix = std::min(static_cast<int>(std::floor(((photon.pos.x - min_x) / length_x) * N_x)), N_x - 1);
    const int iy = std::min(static_cast<int>(std::floor(((photon.pos.y - min_y) / length_y) * N_y)), N_y - 1);

    if (ix < 0 || ix >= N_x || iy < 0 || iy >= N_y)
      return;

    // Compute intensity contribution
    double w = photon.weight;

    Vec3 qb = (photon.s_n + photon.s_0) * photon.k;
    Vec3 delta_r = photon.r_n - photon.r_0;
    std::complex<double> path_phase = std::exp(std::complex<double>(0, dot(qb, delta_r)));
    std::complex<double> phase = std::exp(std::complex<double>(0, photon.k * photon.opticalpath));

    CVec2 E_forw_photon = photon.polarization;
    E_forw_photon.m *= phase * std::sqrt(w);
    E_forw_photon.n *= phase * std::sqrt(w);
    CVec2 E_rev_photon = photon.polarization_reverse;
    E_rev_photon.m *= phase * path_phase * std::sqrt(w);
    E_rev_photon.n *= phase * path_phase * std::sqrt(w);

    // Accumulate coherent intensity contributions
    I_x(ix, iy) += std::norm(E_forw_photon.m * photon.m.x + E_forw_photon.n * photon.n.x + E_rev_photon.m * photon.m.x + E_rev_photon.n * photon.n.x);
    I_y(ix, iy) += std::norm(E_forw_photon.m * photon.m.y + E_forw_photon.n * photon.n.y + E_rev_photon.m * photon.m.y + E_rev_photon.n * photon.n.y);
    I_z(ix, iy) += std::norm(E_forw_photon.m * photon.m.z + E_forw_photon.n * photon.n.z + E_rev_photon.m * photon.m.z + E_rev_photon.n * photon.n.z);

    // Accumulate incoherent intensity contributions
    I_inco_x(ix, iy) += std::norm(E_forw_photon.m * photon.m.x + E_forw_photon.n * photon.n.x) + std::norm(E_rev_photon.m * photon.m.x + E_rev_photon.n * photon.n.x);
    I_inco_y(ix, iy) += std::norm(E_forw_photon.m * photon.m.y + E_forw_photon.n * photon.n.y) + std::norm(E_rev_photon.m * photon.m.y + E_rev_photon.n * photon.n.y);
    I_inco_z(ix, iy) += std::norm(E_forw_photon.m * photon.m.z + E_forw_photon.n * photon.n.z) + std::norm(E_rev_photon.m * photon.m.z + E_rev_photon.n * photon.n.z);

    // LLOG_INFO("Photon hit at ix: {}, iy: {}", ix, iy);

    // Accumulate theta-resolved intensities
    const auto dir = photon.dir;
    const double theta = std::acos(-dir.z);
    const int theta_bin = static_cast<int>(std::floor((theta / (0.05)) * N_x));
    // LLOG_INFO("Theta bin: {}", theta_bin);

    if (theta_bin >= 0 && theta_bin < N_x)
    {
      I_x_theta[theta_bin] += std::norm(E_forw_photon.m * photon.m.x + E_forw_photon.n * photon.n.x + E_rev_photon.m * photon.m.x + E_rev_photon.n * photon.n.x);
      I_y_theta[theta_bin] += std::norm(E_forw_photon.m * photon.m.y + E_forw_photon.n * photon.n.y + E_rev_photon.m * photon.m.y + E_rev_photon.n * photon.n.y);
      I_z_theta[theta_bin] += std::norm(E_forw_photon.m * photon.m.z + E_forw_photon.n * photon.n.z + E_rev_photon.m * photon.m.z + E_rev_photon.n * photon.n.z);

      I_inco_x_theta[theta_bin] += std::norm(E_forw_photon.m * photon.m.x + E_forw_photon.n * photon.n.x) + std::norm(E_rev_photon.m * photon.m.x + E_rev_photon.n * photon.n.x);
      I_inco_y_theta[theta_bin] += std::norm(E_forw_photon.m * photon.m.y + E_forw_photon.n * photon.n.y) + std::norm(E_rev_photon.m * photon.m.y + E_rev_photon.n * photon.n.y);
      I_inco_z_theta[theta_bin] += std::norm(E_forw_photon.m * photon.m.z + E_forw_photon.n * photon.n.z) + std::norm(E_rev_photon.m * photon.m.z + E_rev_photon.n * photon.n.z);
    }
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

      I_inco_x_theta[ix] += other_spatial.I_inco_x_theta[ix];
      I_inco_y_theta[ix] += other_spatial.I_inco_y_theta[ix];
      I_inco_z_theta[ix] += other_spatial.I_inco_z_theta[ix];
      I_x_theta[ix] += other_spatial.I_x_theta[ix];
      I_y_theta[ix] += other_spatial.I_y_theta[ix];
      I_z_theta[ix] += other_spatial.I_z_theta[ix];
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
