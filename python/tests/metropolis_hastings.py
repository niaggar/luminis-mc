#%%
from luminis_mc import MetropolisHastings  
from luminis_mc import ( # Import target distributions
    TargetDistribution,
    HardSpheres,
    Exponential
)
import matplotlib.pyplot as plt
import numpy as np
#==================================================================
# Example usage
#==================================================================


# %%
exp_target_dist = Exponential(1.0) # Create an exponential target distribution with lambda=1.0
HS_target_dist = HardSpheres(1.0, 0.1) # Create a hard spheres target distribution with radius=0.5 and density=0.1

Met_hastings_exp = MetropolisHastings(exp_target_dist) # Create a Metropolis-Hastings sampler
Met_hastings_hs = MetropolisHastings(HS_target_dist) # Create a Metropolis-Hastings sampler for hard spheres

# %%
# Generate 10000 samples starting from initial value 1.0 and proposal standard deviation 0.5
Met_hastings_hs.sample(5000, 1.0, 0.2, positive_support=False)
# %%

# store the samples after burn-in
samples = Met_hastings_hs.MCMC_samples

# Discard the first 500 samples (10%) as burn-in
cleaned_samples = samples[500:]

plt.hist(cleaned_samples, bins=30, density=True)
plt.show()
# %%
