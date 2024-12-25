'''
Module to simulate synthetic data for testing inference methods.

Simulate stochastic reaction network models using the gillespie algorithm to
produce stationary samples emulating single-cell RNA sequencing observations.
Simulate datatsets of samples from large numbers of genes by sampling model
parameters from distributions fitted to match observed data.

Typical example:

# create generator with fixed seed
rng = np.random.default_rng(200)

# set model parameters
params = {
    'k_tx_1': 1,
    'k_tx_2': 1,
    'k_deg_1': 1,
    'k_deg_2': 1,
    'k_reg': 1
}

# simulate sample
sample = gillespie(rng, params, 1000, 0.5)
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import numpy as np
import scipy
import matplotlib.pyplot as plt

# ------------------------------------------------
# Functions
# ------------------------------------------------

def gillespie(rng, params, n, beta, tmax=100, ts=10, plot=False, initial_state=(0, 0)):
    '''
    Simulate a sample path of the birth death regulation model
    Sample n values at intervals of ts after a burn-in time of tmax

    params: dict of reaction rate constants
    n: number of samples
    beta: capture efficiency, list of length n (per cell) or single value
    tmax: burn in time
    ts: time between samples
    '''

    # initialise time and state
    t = 0
    path = [initial_state]
    jump_times = [0]

    # simulate for burn-in time and time between n samples
    while t < tmax + (n - 1) * ts:

        # current state
        x1, x2 = path[-1][0], path[-1][1]

        # transition rates
        q_tx_1 = params['k_tx_1']
        q_tx_2 = params['k_tx_2']
        q_deg_1 = x1 * params['k_deg_1']
        q_deg_2 = x2 * params['k_deg_2']
        q_reg = x1 * x2 * params['k_reg']
        q_hold = q_tx_1 + q_tx_2 + q_deg_1 + q_deg_2 + q_reg

        # holding time in current state
        t_hold = -np.log(rng.uniform()) / q_hold
        t += t_hold
        jump_times.append(t)

        # jump probability
        outcome = [1, 2, 3, 4, 5]
        prob = [
            q_tx_1 / q_hold,
            q_tx_2 / q_hold,
            q_deg_1 / q_hold,
            q_deg_2 / q_hold,
            q_reg / q_hold
        ]
        jump = rng.choice(outcome, p=prob)
        match jump:
            case 1:
                path.append((x1 + 1, x2))
            case 2:
                path.append((x1, x2 + 1))
            case 3:
                path.append((x1 - 1, x2))
            case 4:
                path.append((x1, x2 - 1))
            case 5:
                path.append((x1 - 1, x2 - 1))

    # take the transcript states
    x1_path = [state[0] for state in path]
    x2_path = [state[1] for state in path]

    # create step function of sample path from jump times and jump values
    x1_path_function = scipy.interpolate.interp1d(jump_times, x1_path, kind='previous')
    x2_path_function = scipy.interpolate.interp1d(jump_times, x2_path, kind='previous')

    # take values at sampling times as samples from stationary dist
    sample_times = [tmax + i * ts for i in range(n)]
    x1_samples = x1_path_function(sample_times)
    x2_samples = x2_path_function(sample_times)

    # convert to integers
    x1_samples = [int(x1) for x1 in x1_samples]
    x2_samples = [int(x2) for x2 in x2_samples]

    # apply capture efficiency: for each count, draw from Binomial(count, beta)
    x1_samples_beta = np.random.binomial(x1_samples, beta).tolist()
    x2_samples_beta = np.random.binomial(x2_samples, beta).tolist()

    # re-combine to pairs of samples
    samples = list(zip(x1_samples, x2_samples))
    samples_beta = list(zip(x1_samples_beta, x2_samples_beta))

    # plot sample paths
    if plot:
        x = np.linspace(0, tmax + (n - 1) * ts, 10000)
        plt.plot(x, x1_path_function(x), label="X1 sample path", color="blue")
        plt.plot(x, x2_path_function(x), label="X2 sample path", color="purple")
        #plt.axvline(tmax, label="Burn-in time", color="orange")
        plt.xlabel("Time")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()

    # collect all sample paths: original and observed
    data = {
        'x1_OG': x1_samples,
        'x2_OG': x2_samples,
        'OG': samples,
        'x1_OB': x1_samples_beta,
        'x2_OB': x2_samples_beta,
        'OB': samples_beta
    }

    return data