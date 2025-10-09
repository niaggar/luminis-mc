#%%
from luminis_mc import TargetDistribution, MetropolisHastings, ExpDistribution 
import matplotlib.pyplot as plt
#==================================================================
# Example usage
#==================================================================


# %%
target_distribution = ExpDistribution(1.0) # Define an exponential target distribution with lambda=1.0

Met_hastings_exp = MetropolisHastings(target_distribution) # Create a Metropolis-Hastings sampler

# %%
# Generate 5000 samples starting from initial value 1.0 and proposal standard deviation 0.5

Met_hastings_exp.sample(5000, 1.0, 0.5)
# %%

# store the samples after burn-in
samples = Met_hastings_exp.MCMC_samples;
# Discard the first 500 samples (10%) as burn-in

cleaned_samples = samples[500:];

 # Discard the first 500 samples (10%) as burn-in 

plt.hist(cleaned_samples, bins=30, density=True)
# %%
