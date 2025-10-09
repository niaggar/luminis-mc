#include <cmath>
#include <cstdint>
#include <random>

#include <luminis/math/rng.hpp>


double metropolis_hastings_sample(double (*target_pdf)(double), double x0, double proposal_stddev, std::size_t n_samples, Rng &rng) {


    double current_x = x0;
    double current_pdf = target_pdf(current_x);

    for (std::size_t i = 0; i < n_samples; ++i) {
        // Propose a new sample from a normal distribution centered at current_x
        double proposed_x = current_x + rng.normal(0.0, proposal_stddev);
        double proposed_pdf = target_pdf(proposed_x);

        // Calculate acceptance ratio
        double acceptance_ratio = proposed_pdf / current_pdf;

        // Accept or reject the proposed sample
        if (acceptance_ratio >= 1.0 || rng.uniform() < acceptance_ratio) {
            current_x = proposed_x;
            current_pdf = proposed_pdf;
        }
    }

    return current_x;
}