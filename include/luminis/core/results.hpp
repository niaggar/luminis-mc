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
    std::vector<StokesMatrixProcessed> coherent;
    std::vector<StokesMatrixProcessed> incoherent;
  };

  struct PlanarFluenceProcessed
  {
    std::vector<Matrix> S0, S1, S2, S3; // [N_x x N_y]
  };

  struct PlanarFieldProcessed
  {
    CMatrix Ex, Ey; // [N_x x N_y]
  };

  struct FarFieldCBSRadialProcessed
  {
    std::vector<double> theta_center;
    StokesRadialProcessed coherent;
    StokesRadialProcessed incoherent;
  };

  PlanarFluenceProcessed postprocess_planar_fluence(const PlanarFluenceSensor &det, std::size_t n_photons, bool normalize_per_photon = true, bool normalize_per_area = true, double eps = 1e-30);
  PlanarFieldProcessed postprocess_planar_field(const PlanarFieldSensor &det, std::size_t n_photons, bool normalize_per_photon = true, bool normalize_per_area = true, double eps = 1e-30);
  FarFieldCBSProcessed postprocess_farfield_cbs(const FarFieldCBSSensor &det, std::size_t n_photons, bool normalize_per_solid_angle = true, bool normalize_per_photon = true, double eps = 1e-30);
  // FarFieldCBSRadialProcessed radial_average_S0(const FarFieldCBSSensor &det, std::size_t n_photons, bool normalize_per_solid_angle = true, bool normalize_per_photon = true, double eps = 1e-30);
}