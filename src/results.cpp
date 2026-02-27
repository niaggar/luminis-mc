#pragma once
#include <luminis/core/results.hpp>

namespace luminis::core
{
  FarFieldCBSProcessed postprocess_farfield_cbs(const FarFieldCBSSensor &det, std::size_t n_photons, bool normalize_per_solid_angle, bool normalize_per_photon, double eps)
  {
    Matrix dOmega(det.N_theta, det.N_phi);
    std::vector<StokesMatrixProcessed> coherent(det.N_t);
    std::vector<StokesMatrixProcessed> incoherent(det.N_t);
    for (int t = 0; t < det.N_t; ++t)
    {
      coherent[t] = {Matrix(det.N_theta, det.N_phi), Matrix(det.N_theta, det.N_phi), Matrix(det.N_theta, det.N_phi), Matrix(det.N_theta, det.N_phi)};
      incoherent[t] = {Matrix(det.N_theta, det.N_phi), Matrix(det.N_theta, det.N_phi), Matrix(det.N_theta, det.N_phi), Matrix(det.N_theta, det.N_phi)};
    }

    const double invN = (normalize_per_photon && n_photons > 0) ? (1.0 / double(n_photons)) : 1.0;

    for (int i = 0; i < det.N_theta; ++i)
    {
      const double dOm = normalize_per_solid_angle ? solid_angle_bin(i, det.dtheta, det.dphi) : 1.0;
      const double invOm = (dOm > eps) ? (1.0 / dOm) : 0.0;

      for (int j = 0; j < det.N_phi; ++j)
      {
        dOmega(i, j) = dOm;

        const double norm = invN * invOm;

        for (int t = 0; t < det.N_t; ++t)
        {
          coherent[t].S0(i, j) = det.S0_coh[t](i, j) * norm;
          coherent[t].S1(i, j) = det.S1_coh[t](i, j) * norm;
          coherent[t].S2(i, j) = det.S2_coh[t](i, j) * norm;
          coherent[t].S3(i, j) = det.S3_coh[t](i, j) * norm;

          incoherent[t].S0(i, j) = det.S0_incoh[t](i, j) * norm;
          incoherent[t].S1(i, j) = det.S1_incoh[t](i, j) * norm;
          incoherent[t].S2(i, j) = det.S2_incoh[t](i, j) * norm;
          incoherent[t].S3(i, j) = det.S3_incoh[t](i, j) * norm;
        }
      }
    }

    return FarFieldCBSProcessed{dOmega, coherent, incoherent};
  }


}