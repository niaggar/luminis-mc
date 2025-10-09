#pragma once
#include <cmath>
#include <vector>
#include <luminis/math/rng.hpp>

namespace luminis::sample {

class TargetDistribution {
public:
    virtual ~TargetDistribution() = default;
    virtual double evaluate(double x) = 0;
};

class metropolis_hastings {
private:
    TargetDistribution *target_dist{nullptr};
    double proposal_distribution(double x, double stddev);    
    int acceptance_count = 0;

public:
    metropolis_hastings(TargetDistribution *target_dist) {
        this->target_dist = target_dist;
    }
    
    void accept_reject(double &current, double &target_dist_current, double proposal_stddev);
    void sample(int num_samples, double initial_value, double proposal_stddev);
    std::vector<double> MCMC_samples; 

};


class ExpDistribution : public TargetDistribution {
public:
    ExpDistribution(double lambda);
    double evaluate(double x) override;
private:
    double lambda; // mean free path
};

}