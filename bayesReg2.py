# 181206 created bayesReg2.py
# derived from https://www.quantstart.com/articles/Bayesian-Linear-Regression-Models-with-PyMC3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import pymc3 as pm
from pymc3 import  *


sns.set(style="darkgrid", palette="muted")


def simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq):
    """
    Simulate a random dataset using a noisy
    linear process.

    N: Number of data points to simulate
    beta_0: Intercept
    beta_1: Slope of univariate predictor, X
    """
    # Create a pandas DataFrame with column 'x' containing
    # N uniformly sampled values between 0.0 and 1.0
    df = pd.DataFrame(
        {"x": 
            np.random.RandomState(42).choice(
                list(map(
                    lambda x: float(x)/100.0, 
                    np.arange(100)
                )), N, replace=False
            )
        }
    )

    # Use a linear model (y ~ beta_0 + beta_1*x + epsilon) to 
    # generate a column 'y' of responses based on 'x'
    eps_mean = 0.0
    df["y"] = beta_0 + beta_1*df["x"] + np.random.RandomState(42).normal(
        eps_mean, eps_sigma_sq, N
    )

    return df

def glm_mcmc_inference(df, iterations=5000):
    """
    Calculates the Markov Chain Monte Carlo trace of
    a Generalised Linear Model Bayesian linear regression 
    model on supplied data.

    df: DataFrame containing the data
    iterations: Number of iterations to carry out MCMC for
    """
    # Use PyMC3 to construct a model context
    # basic_model = pm.Model()
    with Model() as model:
        # Create the glm using the Patsy model syntax
        # We use a Normal distribution for the likelihood
        glm.GLM.from_formula("y ~ x", df)
       
        # Use Maximum A Posteriori (MAP) optimisation as initial value for MCMC
        # start = pm.find_MAP()

        # Use the No-U-Turn Sampler
        # step = pm.NUTS()

        # Calculate the trace
        trace = sample(3000)

    return trace


if __name__ == "__main__":
    # These are our "true" parameters
    beta_0 = 1.0  # Intercept
    beta_1 = 2.0  # Slope

    # Simulate 100 data points, with a variance of 0.5
    N = 100
    eps_sigma_sq = 0.5

    # Simulate the "linear" data using the above parameters
    df = simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq)

    # Plot the data, and a frequentist linear regression fit
    # using the seaborn package
    sns.lmplot(x="x", y="y", data=df, height=10)
    plt.xlim(0.0, 1.0)
    plt.show()
    trace = glm_mcmc_inference(df, iterations=5000)
    plt.figure(figsize=(7, 7))
    traceplot(trace[100:])
    plt.tight_layout();
    plt.show()
