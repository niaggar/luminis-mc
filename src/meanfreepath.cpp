#include <iostream>
#include <luminis/sample/meanfreepath.hpp>
#include <luminis/math/rng.hpp>
#include <cmath>

using luminis::math::Rng; 

namespace luminis::sample {


double metropolis_hastings::proposal_distribution(double x, double stddev){ 
    luminis::math::Rng rng;
    return rng.normal(x, stddev);
}

void metropolis_hastings::accept_reject(double &current, double &target_dist_current, double proposal_stddev) {

    luminis::math::Rng rng;

    double proposed = proposal_distribution(current, proposal_stddev);
    double current_deprecated = current;

    double target_dist_proposed = target_dist->evaluate(proposed);

    double acceptance_ratio = target_dist_proposed / target_dist_current;

    double acceptance_prob = std::min(1.0, acceptance_ratio);

    if (rng.uniform() < acceptance_prob) {
        current = proposed;
        target_dist_current = target_dist_proposed;
        acceptance_count++;
    }

}

void metropolis_hastings::sample(int num_samples, double initial_value, double proposal_stddev)
{

    std::vector<double> samples;
    samples.clear();

    double current = initial_value;
    double target_dist_current = target_dist->evaluate(current);

    std::vector<double> draws(num_samples, 0.0);

    for (int i = 0; i < num_samples; i++) {

        metropolis_hastings::accept_reject(current, target_dist_current, proposal_stddev);
        samples.push_back(current);
    }

    std::cout << "Acceptance rate: " << acceptance_count / static_cast<double>(num_samples) << "%" << std::endl;

    acceptance_count = 0;
    MCMC_samples = samples;
}


ExpDistribution::ExpDistribution(double lambda) {
    this->lambda = lambda;
}

double ExpDistribution::evaluate(double x) {
    if (x < 0.0) return 0.0; // Exponential distribution is defined for x >= 0
    return  std::exp(-lambda * x);

} 
} // namespace luminis::sample
