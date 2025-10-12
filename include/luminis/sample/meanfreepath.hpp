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
    int acceptance_count = 0;

public:
    metropolis_hastings(TargetDistribution *target_dist) {
        this->target_dist = target_dist;
    }

    void accept_reject(double &current, double &target_dist_current, double proposal_stddev, bool positive_support);
    void sample(int num_samples, double initial_value, double proposal_stddev, bool positive_support);
    std::vector<double> MCMC_samples; 

};


class Exponential : public TargetDistribution {
public:
    Exponential(double lambda);
    double evaluate(double x) override;
private:
    double lambda; // mean free path
};

class HardSpheres : public TargetDistribution {
public:
    HardSpheres(double radius, double density);
    double evaluate(double x) override;
private:
    double radius; // radius of the hard sphere
    double density; // density of the hard sphere
};


}