/**
 * @file meanfreepath.hpp
 * @brief Metropolis-Hastings sampler for free-path distributions.
 *
 * Provides a generic MCMC sampler (`metropolis_hastings`) that draws samples
 * from a user-supplied `TargetDistribution`. Two concrete targets are included:
 * an `Exponential` free-path law and a `HardSpheres` structure-factor-corrected
 * distribution for dense suspensions.
 */

#pragma once
#include <cmath>
#include <vector>
#include <luminis/math/rng.hpp>

namespace luminis::sample {

/// @brief Abstract (unnormalized) target distribution to be sampled.
class TargetDistribution {
public:
    virtual ~TargetDistribution() = default;
    /// @brief Evaluate the unnormalized density at `x`.
    virtual double evaluate(double x) = 0;
};

/// @brief Metropolis-Hastings MCMC sampler over a TargetDistribution.
class metropolis_hastings {
private:
    TargetDistribution *target_dist{nullptr}; ///< Non-owning pointer to the target density.
    int acceptance_count = 0;                 ///< Number of accepted proposals (for the acceptance rate).

public:
    /// @param target_dist Density to sample from (must outlive this sampler).
    metropolis_hastings(TargetDistribution *target_dist) {
        this->target_dist = target_dist;
    }

    /// @brief Perform one accept/reject step with a Gaussian proposal.
    /// @param current               Current state (updated in place if accepted).
    /// @param target_dist_current   Density at the current state (updated in place).
    /// @param proposal_stddev       Standard deviation of the Gaussian proposal.
    /// @param positive_support      If true, proposals outside x > 0 are rejected.
    void accept_reject(double &current, double &target_dist_current, double proposal_stddev, bool positive_support);

    /// @brief Run the chain and store the draws in `MCMC_samples`.
    /// @param num_samples      Number of samples to generate.
    /// @param initial_value    Starting state of the chain.
    /// @param proposal_stddev  Standard deviation of the Gaussian proposal.
    /// @param positive_support Restrict the chain to positive values.
    void sample(int num_samples, double initial_value, double proposal_stddev, bool positive_support);

    std::vector<double> MCMC_samples; ///< Generated samples (filled by sample()).
};

/// @brief Exponential free-path distribution p(x) ∝ exp(-x/λ).
class Exponential : public TargetDistribution {
public:
    /// @param lambda Mean free path λ.
    Exponential(double lambda);
    double evaluate(double x) override;
private:
    double lambda; ///< Mean free path.
};

/// @brief Hard-sphere structure-factor-corrected free-path distribution.
class HardSpheres : public TargetDistribution {
public:
    /// @param radius  Hard-sphere radius.
    /// @param density Number density of spheres.
    HardSpheres(double radius, double density);
    double evaluate(double x) override;
private:
    double radius;  ///< Hard-sphere radius.
    double density; ///< Number density of spheres.
};

}
