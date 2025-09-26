#include <cmath>
#include <luminis/core/simulation.hpp>
#include <luminis/log/logger.hpp>

namespace luminis::core
{

    void run_simulation(const SimConfig &config, Medium &medium, Detector &detector)
    {
        Rng rng(config.seed);
    
        for (std::size_t i = 0; i < config.n_photons; ++i)
        {
        Photon photon;
        run_photon(photon, medium, detector);
        }
    }
    
    void run_photon(Photon &photon, Medium &medium, Detector &detector)
    {
        Rng rng; // Local RNG for each photon

        for (std::size_t scatter_count = 0; scatter_count < 1000; ++scatter_count)
        {
            // Sample free path
            double path = medium.sample_free_path(rng);
            photon.pos = photon.pos + photon.dir * path;
    
            
            // Check for detector hit
    

            // Absorption check
    

            // Scatter the photon
            double theta = medium.sample_scattering_angle(rng);
            double phi = medium.sample_azimuthal_angle(rng);


            // Get scattering matrix


            // Update photon polarization if needed


            // Update photon direction



            photon.weight = photon.weight - (medium.mu_a / medium.mu_s) * photon.weight;
            photon.opticalpath += path;
            photon.events++;
        }
    }

}
