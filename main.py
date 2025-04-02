import compute
import os

# Given parameters
sizes = [20, 60, 100]
rhos = [0, 0.5, 0.9]

# Computing and printing results
normal_results = compute.generate_samples(sizes, rhos)
mixture_results = compute.generate_mixture_samples(sizes)

print("\nVisualizing...")
# Create a folder for plots, if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

for size in sizes:
    for rho in rhos:
        compute.plot_samples_and_ellipse(size, rho)

print("\nPlots are saved in 'plots' folder")
