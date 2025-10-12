#include <iostream>
#include <luminis/sample/meanfreepath.hpp>
#include <luminis/math/rng.hpp>
#include <cmath>

using luminis::math::Rng; 

namespace luminis::sample {


void metropolis_hastings::accept_reject(double &current, double &target_dist_current, double proposal_stddev, bool positive_support) {

    luminis::math::Rng rng;

    double acceptance_ratio = 0.0;
    double proposed = 0.0;
    double target_dist_proposed = 0.0;

    if (positive_support == true) { 

        double log_proposed = std::log(current) + rng.normal(0.0, proposal_stddev); // Log-normal proposal distribution for positive support

        proposed = std::exp(log_proposed);
        target_dist_proposed = target_dist->evaluate(proposed);

        double log_acceptance_ratio = std::log(target_dist_proposed) - std::log(target_dist_current) + (log_proposed - std::log(current));

        acceptance_ratio = std::exp(log_acceptance_ratio);

    } else if (positive_support == false) {

        proposed = current + rng.normal(0.0, proposal_stddev);
        target_dist_proposed = target_dist->evaluate(proposed);
        acceptance_ratio = target_dist_proposed / target_dist_current;
    } else {
        throw std::invalid_argument("positive_support must be true or false");
    }


    if (rng.uniform() < acceptance_ratio) {
        current = proposed;
        target_dist_current = target_dist_proposed;
        acceptance_count++;
    }

}

void metropolis_hastings::sample(int num_samples, double initial_value, double proposal_stddev, bool positive_support)
{

    std::vector<double> samples;
    samples.clear();

    double current = initial_value;
    double target_dist_current = target_dist->evaluate(current);

    std::vector<double> draws(num_samples, 0.0);

    for (int i = 0; i < num_samples; i++) {

        metropolis_hastings::accept_reject(current, target_dist_current, proposal_stddev, positive_support);
        samples.push_back(current);
    }

    std::cout << "Acceptance rate: " << acceptance_count / static_cast<double>(num_samples) << "%" << std::endl;

    acceptance_count = 0;
    MCMC_samples = samples;
}

//========================================
// Mean free path target distributions
//========================================


Exponential::Exponential(double lambda) {
    this->lambda = lambda;
}

double Exponential::evaluate(double x) {
    if (x < 0.0) return 0.0; // Exponential distribution is defined for x >= 0
    return  std::exp(-lambda * x);

} 

HardSpheres::HardSpheres(double radius, double density) {
    this->radius = radius;
    this->density = density;
}

double HardSpheres::evaluate(double x) {
    if (x < 0.0) return 0.0; // Hard sphere distribution is defined for x >= 0

    if (x < radius && x >= 0.0) {
        return 1.0; 
    } else if (x >= radius) {
        return std::exp(-4.0*(M_PI/3.0) * density * std::pow(x, 3));
    } 
}

} // namespace luminis::sample
