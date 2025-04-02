import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from matplotlib.patches import Ellipse


def generate_samples(sizes, rhos, n_iterations=1000):
    """
    Generates samples and calculates statistics for correlation coefficients.
    Outputs results as LaTeX tables to the console.
    """
    for size in sizes:
        # Begin LaTeX table for current sample size
        print(f"% Results for sample size n = {size}")
        print("\\begin{table}[H]")
        print("\\centering")
        print("\\begin{tabular}{|c|c|c|c|c|c|c|}")
        print("\\hline")
        print("$\\rho$ & Method & Mean & Mean of squares & Variance \\\\")
        print("\\hline")

        for rho in rhos:
            # Initialize vectors for storing results
            pearson_vals = np.zeros(n_iterations)
            spearman_vals = np.zeros(n_iterations)
            quadratic_vals = np.zeros(n_iterations)

            # Covariance matrix for normal distribution
            cov = np.array([[1, rho], [rho, 1]])

            for i in range(n_iterations):
                # Generate normal distribution
                data = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=size)

                # Calculate correlation coefficients
                pearson_r, _ = pearsonr(data[:, 0], data[:, 1])
                spearman_r, _ = spearmanr(data[:, 0], data[:, 1])
                quadratic_r = np.corrcoef(data[:, 0] ** 2, data[:, 1] ** 2)[0, 1]

                pearson_vals[i] = pearson_r
                spearman_vals[i] = spearman_r
                quadratic_vals[i] = quadratic_r

            # Calculate statistics
            pearson_mean = np.mean(pearson_vals)
            pearson_mean_sq = np.mean(pearson_vals ** 2)
            pearson_var = np.var(pearson_vals)

            spearman_mean = np.mean(spearman_vals)
            spearman_mean_sq = np.mean(spearman_vals ** 2)
            spearman_var = np.var(spearman_vals)

            quadratic_mean = np.mean(quadratic_vals)
            quadratic_mean_sq = np.mean(quadratic_vals ** 2)
            quadratic_var = np.var(quadratic_vals)

            # Output results in LaTeX format
            print(f"{rho:.1f} & $Pearson^{{(1)}}$ & {pearson_mean:.4f} & {pearson_mean_sq:.4f} & {pearson_var:.4f} \\\\")
            print("\\hline")
            print(f"{rho:.1f} & $Spearman^{{(2)}}$ & {spearman_mean:.4f} & {spearman_mean_sq:.4f} & {spearman_var:.4f} \\\\")
            print("\\hline")
            print(f"{rho:.1f} & $Quadratic^{{(3)}}$ & {quadratic_mean:.4f} & {quadratic_mean_sq:.4f} & {quadratic_var:.4f} \\\\")
            print("\\hline")

        # End LaTeX table
        print("\\end{tabular}")
        print(f"\\caption{{Statistics for correlation coefficients (n = {size})}}")
        print("\\end{table}")
        print()  # Add empty line between tables


def generate_mixture_samples(sizes, n_iterations=1000):
    """
    Generates samples from mixture distributions and calculates statistics.
     Outputs results as LaTeX tables to the console.
    """

    for size in sizes:
        pearson_vals = np.zeros(n_iterations)
        spearman_vals = np.zeros(n_iterations)
        quadratic_vals = np.zeros(n_iterations)

        cov1 = np.array([[1, 0.9], [0.9, 1]])  # First distribution (90%)
        cov2 = np.array([[10, -0.9], [-0.9, 10]])  # Second distribution (10%)

        for i in range(n_iterations):
            # Generate mixed distribution
            n1 = int(0.9 * size)
            n2 = size - n1

            data1 = np.random.multivariate_normal(mean=[0, 0], cov=cov1, size=n1)
            data2 = np.random.multivariate_normal(mean=[0, 0], cov=cov2, size=n2)
            data = np.vstack((data1, data2))

            # Calculate correlation coefficients
            pearson_r, _ = pearsonr(data[:, 0], data[:, 1])
            spearman_r, _ = spearmanr(data[:, 0], data[:, 1])
            quadratic_r = np.corrcoef(data[:, 0] ** 2, data[:, 1] ** 2)[0, 1]

            pearson_vals[i] = pearson_r
            spearman_vals[i] = spearman_r
            quadratic_vals[i] = quadratic_r

        # Calculate statistics
        pearson_mean = np.mean(pearson_vals)
        pearson_mean_sq = np.mean(pearson_vals ** 2)
        pearson_var = np.var(pearson_vals)

        spearman_mean = np.mean(spearman_vals)
        spearman_mean_sq = np.mean(spearman_vals ** 2)
        spearman_var = np.var(spearman_vals)

        quadratic_mean = np.mean(quadratic_vals)
        quadratic_mean_sq = np.mean(quadratic_vals ** 2)
        quadratic_var = np.var(quadratic_vals)

        # Print LaTeX table
        print(f"\\begin{{table}}[H]")
        print(f"\\centering")
        print(f"\\caption{{Statistics for sample size n = {size}}}")
        print(f"\\begin{{tabular}}{{lccc}}")
        print(f"\\hline")
        print(f"Statistic & $Pearson^{{(1)}}$ & $Spearman^{{(2)}}$ & $Quadratic^{{(3)}}$ \\\\")
        print(f"\\hline")
        print(f"Mean & {pearson_mean:.4f} & {spearman_mean:.4f} & {quadratic_mean:.4f} \\\\")
        print(f"Mean squared & {pearson_mean_sq:.4f} & {spearman_mean_sq:.4f} & {quadratic_mean_sq:.4f} \\\\")
        print(f"Variance & {pearson_var:.4f} & {spearman_var:.4f} & {quadratic_var:.4f} \\\\")
        print(f"\\hline")
        print(f"\\end{{tabular}}")
        print(f"\\end{{table}}")
        print()  # Add empty line between tables


def plot_samples_and_ellipse(size, rho):
    """
    Visualization of samples and equal probability ellipses
    (theoretical and sample-based)
    """
    # Theoretical covariance
    cov_theory = np.array([[1, rho], [rho, 1]])
    data = np.random.multivariate_normal(mean=[0, 0], cov=cov_theory, size=size)

    # Sample covariance
    cov_sample = np.cov(data, rowvar=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(data[:, 0], data[:, 1], alpha=0.6)

    # Draw theoretical ellipse (red)
    lambda_theory, v_theory = np.linalg.eig(cov_theory)
    lambda_theory = np.sqrt(lambda_theory)
    ellipse_theory = Ellipse(xy=(0, 0),
                             width=2 * lambda_theory[0] * 2,
                             height=2 * lambda_theory[1] * 2,
                             angle=np.degrees(np.arccos(v_theory[0, 0])),
                             edgecolor='r', fc='None', lw=2, linestyle='--',
                             label='Theoretical')
    ax.add_patch(ellipse_theory)

    # Draw sample ellipse
    if size > 1:  # Covariance is defined for sample size > 1
        lambda_sample, v_sample = np.linalg.eig(cov_sample)
        lambda_sample = np.sqrt(lambda_sample)
        ellipse_sample = Ellipse(xy=(np.mean(data[:, 0]), np.mean(data[:, 1])),
                                 width=2 * lambda_sample[0] * 2,
                                 height=2 * lambda_sample[1] * 2,
                                 angle=np.degrees(np.arccos(v_sample[0, 0])),
                                 edgecolor='b', fc='None', lw=2,
                                 label='Sample-based')
        ax.add_patch(ellipse_sample)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_title(f'Sample size: {size}, ρ: {rho}')
    ax.legend()
    ax.grid(True)

    filename = f'plots/normal_dist_size_{size}_rho_{rho}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close(fig)
