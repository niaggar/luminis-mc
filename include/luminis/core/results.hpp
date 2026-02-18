#pragma once
#include <cstddef>
#include <luminis/core/detector.hpp>
#include <luminis/core/photon.hpp>
#include <luminis/math/vec.hpp>
#include <luminis/math/utils.hpp>
#include <vector>
#include <functional>
#include <memory>
#include <map>

namespace luminis::core
{
  struct StokesMatrixProcessed
  {
    Matrix S0, S1, S2, S3;
  };

  struct StokesRadialProcessed
  {
    std::vector<double> S0, S1, S2, S3;
  };

  struct FarFieldCBSProcessed
  {
    Matrix dOmega; // [N_theta x N_phi]
    StokesMatrixProcessed coherent;
    StokesMatrixProcessed incoherent;
  };

  struct FarFieldCBSRadialProcessed
  {
    std::vector<double> theta_center;
    StokesRadialProcessed coherent;
    StokesRadialProcessed incoherent;
  };

  FarFieldCBSProcessed postprocess_farfield_cbs(const FarFieldCBSSensor &det, std::size_t n_photons, bool normalize_per_solid_angle = true, bool normalize_per_photon = true, double eps = 1e-30);
  // FarFieldCBSRadialProcessed radial_average_S0(const FarFieldCBSSensor &det, std::size_t n_photons, bool normalize_per_solid_angle = true, bool normalize_per_photon = true, double eps = 1e-30);
}