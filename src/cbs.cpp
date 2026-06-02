// ═══════════════════════════════════════════════════════════════════════════
//  CBS far-field sensor — refactored
//
//  Design goal: the three-stage reverse-path rotation algorithm (Xu 2008,
//  Eq. 3;  E_rev = S^(1) R(φ1') T^rev R(φn') S^(n) R(φn) E0,  T^rev = Q T^T Q)
//  lives in EXACTLY ONE place: reverse_field(). Both the direct detector
//  (process_hit) and the last-flight estimator (process_estimation) call it.
//  They differ only in WHICH frames and WHICH raw interior product they pass.
//  Edit the rotations once and both paths stay consistent.
//
//  IMPORTANT (configuration, not a code bug): process_estimation (last-flight
//  estimator) and process_hit (analog detection) write into the SAME Stokes
//  arrays. They are two mutually-exclusive estimators of the angular
//  distribution — do NOT run both on the same sensor or you double count.
//  Pick one per FarFieldCBSSensor instance.
//
//  Declarations to add to the header:
//    private:
//      const ScatteringMedium* _I_norm_medium = nullptr;   // cache key
//      int  time_bin(double arrival_time) const;
//      void accumulate_stokes(int it, int jp, int t_idx,
//                             std::complex<double> Efx, std::complex<double> Efy,
//                             std::complex<double> Erx, std::complex<double> Ery);
//    // free functions in the .cpp:
//      CVec2 reverse_field(...); static double phase_F(...);
//      static Matrix scatter_frame(...); static void project_to_lab(...);
// ═══════════════════════════════════════════════════════════════════════════

#include <cmath>
#include <complex>
#include <luminis/core/detector.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/core/medium.hpp>
#include <luminis/core/sample.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/math/utils.hpp>
#include <luminis/log/logger.hpp>
#include <vector>
#include <functional>

// ─────────────────────────────────────────────────────────────────────────────
//  Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

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
//  Reverse-path electric field — THE single implementation of the 3-stage algo
//
//  Stage A : first reverse scatter at r_n,   s0      -> -s_{n-1}
//  Stage B : interior segment via reciprocity, E <- Q · T_interior^T · Q · E
//  Stage C : last reverse scatter at r_1,     -s1     ->  s_n
//
//  Uses the RAW (un-F-normalized) interior product and normalizes the result
//  ONCE at the end. Energy is carried by photon.weight, not by the Jones norm,
//  so any scalar inside the path cancels — keeping this consistent with the
//  normalized forward field used by both callers.
// ─────────────────────────────────────────────────────────────────────────────
CVec2 reverse_field(const Sample &medium,
                    const Matrix &P0, const Matrix &P1,
                    const Matrix &P_nm1, const Matrix &P_n,
                    const CMatrix &T_interior_raw,
                    int layer_n, int layer_1,
                    const CVec2 &E_in)
{
  // Propagation directions (row 2) and transverse bases (rows 0,1).
  const Vec3 s0 = row_vec3(P0, 2);
  const Vec3 s1 = row_vec3(P1, 2);
  const Vec3 snm1 = row_vec3(P_nm1, 2);
  const Vec3 sn = row_vec3(P_n, 2);

  const Vec3 m0 = row_vec3(P0, 0), n0 = row_vec3(P0, 1);
  const Vec3 m1 = row_vec3(P1, 0), n1 = row_vec3(P1, 1);
  const Vec3 mnm1 = row_vec3(P_nm1, 0), nnm1 = row_vec3(P_nm1, 1);
  const Vec3 mn = row_vec3(P_n, 0), nn = row_vec3(P_n, 1);

  CMatrix Q(2, 2);
  Q(0, 0) = 1;
  Q(0, 1) = 0;
  Q(1, 0) = 0;
  Q(1, 1) = -1;

  // ── Stage A: scatter at r_n, s0 -> -s_{n-1} ──
  const Vec3 s_in_a = s0;
  const Vec3 s_out_a = snm1 * (-1.0);
  const double th_a = std::acos(clamp_pm1(dot(s_in_a, s_out_a)));
  const CMatrix S_a = medium.get_layer(layer_n).medium->scattering_matrix(th_a, 0.0);

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

  // ── Stage C: scatter at r_1, -s1 -> sn ──
  const Vec3 s_in_c = s1 * (-1.0);
  const Vec3 s_out_c = sn;
  const double th_c = std::acos(clamp_pm1(dot(s_in_c, s_out_c)));
  const CMatrix S_c = medium.get_layer(layer_1).medium->scattering_matrix(th_c, 0.0);

  const Vec3 npp = safe_unit(cross(s_in_c, s_out_c), n1 * (-1.0));
  const Vec3 mpp_in = safe_unit(cross(npp, s_in_c), m1);
  const Vec3 mpp_out = safe_unit(cross(npp, s_out_c), mn);

  CMatrix SR_c(2, 2);
  matcmul(S_c, rot2(mpp_in, npp, m1, n1 * (-1.0)), SR_c); // S(θ)·R(φ1')
  E = apply2(SR_c, E);
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
  if (photon.events < 2) // CBS needs ≥2 events to form a time-reversed pair.
    return;

  const int t_idx = time_bin(photon.launch_time + photon.opticalpath / photon.velocity);
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
      medium, photon.P0, photon.P1, photon.Pn1, photon.Pn,
      photon.matrix_T_raw, photon.last_scatter_layer, photon.first_scatter_layer,
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
  if (photon.events < 1) // estimated path = events+1 ≥ 2 scatters.
    return;

  const ScatteringMedium *med = medium.get_layer(photon.current_layer).medium;

  // I_norm cache: keyed on BOTH medium and k (critical for layered samples).
  if (_I_norm < 0.0 || _I_norm_medium != med || std::abs(_I_norm_k - photon.k) > 1e-12)
  {
    _I_norm = compute_I_norm(*med, photon.k);
    _I_norm_k = photon.k;
    _I_norm_medium = med;
  }
  if (_I_norm < 1e-300)
    return;

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

  for (int it = 0; it < i_max; ++it)
  {
    const double th0 = it * dtheta;
    const double th1 = (it + 1) * dtheta;
    const double dOmega_theta = std::cos(th0) - std::cos(th1); // exact θ-band solid angle (×dφ)
    const double th_det = (it + 0.5) * dtheta;
    const double s_th = std::sin(th_det);
    const double c_th = std::cos(th_det);

    for (int jp = 0; jp < N_phi; ++jp)
    {
      const double ph_det = (jp + 0.5) * dphi;
      const double c_ph = std::cos(ph_det);
      const double s_ph = std::sin(ph_det);

      // Bin direction in lab coords: s_out = sinθ cosφ e1 + sinθ sinφ e2 + cosθ e3.
      const Vec3 s_out = e1 * (s_th * c_ph) + e2 * (s_th * s_ph) + e3 * c_th;

      // Path to the detector plane and transmittance without further scatter.
      double L;
      if (!intersect_plane(photon.pos, s_out, origin, normal, L))
        continue;
      const double Tr = std::exp(-med->mu_attenuation * L);
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
      const double prob_bin = (F / (M_PI * _I_norm)) * (dOmega_theta * dphi);
      const double w_bin = w_scatter * Tr * prob_bin;
      if (w_bin < 1e-300)
        continue;

      const std::complex<double> amp =
          std::sqrt(w_bin) * std::exp(std::complex<double>(0.0, photon.k * L));

      // Estimated exit frame (same convention as transport).
      const Matrix P_exit = scatter_frame(Pcur, th_scat, cos_phi, sin_phi);

      // Forward (unit) and reverse (unit) Jones vectors in the P_exit basis.
      const CVec2 Ef_loc = apply_scatter_normalized(S, cos_phi, sin_phi, photon.polarization);
      const CVec2 Er_loc = reverse_field(
          medium, photon.P0, photon.P1, Pcur, P_exit,
          Tmid_raw, photon.current_layer, photon.first_scatter_layer,
          photon.initial_polarization);

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