'''
Module to compute state space truncations and coefficients for use
when constructing probability optimization constraints.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy
import tqdm

# ------------------------------------------------
# Display observed truncation
# ------------------------------------------------

def illustrate_truncation(truncation_OB, truncationM_OB):
    rng = np.random.default_rng()
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    for i, truncation in truncation_OB.items():
        colour = list(rng.integers(0, 256, size=3))
        axs[0].hlines([truncation['min_x1_OB'] - 0.5, truncation['max_x1_OB'] + 0.5], xmin=truncation['min_x2_OB'] - 0.5, xmax=truncation['max_x2_OB'] + 0.5, color=[colour, colour], linewidth=2)
        axs[0].vlines([truncation['min_x2_OB'] - 0.5, truncation['max_x2_OB'] + 0.5], ymin=truncation['min_x1_OB'] - 0.5, ymax=truncation['max_x1_OB'] + 0.5, color=[colour, colour], linewidth=2)
        axs[0].set_title("OB truncation")
    for i, truncation in truncationM_OB.items():
        colour = list(rng.integers(0, 256, size=3))
        axs[1].hlines([truncation['minM_x1_OB'] - 0.5, truncation['maxM_x1_OB'] + 0.5], xmin=truncation['minM_x2_OB'] - 0.5, xmax=truncation['maxM_x2_OB'] + 0.5, color=[colour, colour], linewidth=2)
        axs[1].vlines([truncation['minM_x2_OB'] - 0.5, truncation['maxM_x2_OB'] + 0.5], ymin=truncation['minM_x1_OB'] - 0.5, ymax=truncation['maxM_x1_OB'] + 0.5, color=[colour, colour], linewidth=2)
        axs[1].set_title("OB marginal truncation")
    plt.show()

# ------------------------------------------------
# Summarise observed truncation
# ------------------------------------------------

def summarise_truncation(truncation_OB, truncationM_OB):
    '''
    Summarise states included in collection of observed truncations
    '''

    # state set
    state_pairs = set()
    states = set()

    # loop over each truncation
    for i, trunc in truncation_OB.items():

        # for each state pair in truncation
        for x1_OB in range(trunc['min_x1_OB'], trunc['max_x1_OB'] + 1):
            for x2_OB in range(trunc['min_x2_OB'], trunc['max_x2_OB'] + 1):

                # add to set
                state_pairs.add((x1_OB, x2_OB))
                states.add(x1_OB)
                states.add(x2_OB)

    # also add any single states (not pairs) in marginal truncations that were missed
    for i, trunc in truncationM_OB.items():
        for x1_OB in range(trunc['minM_x1_OB'], trunc['maxM_x1_OB'] + 1):
            states.add(x1_OB)
        for x2_OB in range(trunc['minM_x2_OB'], trunc['maxM_x2_OB'] + 1):
            states.add(x2_OB)

    # collect info
    info_dict = {
        'state_pairs': state_pairs,
        'states': states
    }

    return info_dict

# ------------------------------------------------
# Compute original truncation
# ------------------------------------------------

def Bm_trunc(x_OB, x_OG, beta):
    return np.mean(scipy.stats.binom.pmf(x_OB, x_OG, beta))

def marginal_truncation(x_OB, beta, thresh_OG=10**-6):

    # start at first non-zero coefficient
    x_OG = x_OB
    coeff = Bm_trunc(x_OB, x_OG, beta)

    # if not above threshold: increment until above
    while coeff < thresh_OG:

        # increment
        x_OG += 1

        # compute coeff
        coeff = Bm_trunc(x_OB, x_OG, beta)

    # store first state coeff >= thresh
    minM_OG = x_OG

    # increment until below threshold
    while coeff >= thresh_OG:

        # increment
        x_OG += 1

        # compute coeff
        coeff = Bm_trunc(x_OB, x_OG, beta)

    # store last state with coeff >= thresh (INCLUSIVE BOUND)
    maxM_OG = x_OG - 1

    return minM_OG, maxM_OG

def original_truncation(truncation_summary, beta, thresh_OG=10**-6, tqdm_disable=True):
    
    # collect OG truncations
    truncation_dict = {}

    # compute truncation for each observed count
    for x_OB in tqdm.tqdm(truncation_summary['states'], disable=tqdm_disable):
        
        minM_OG, maxM_OG = marginal_truncation(x_OB, beta, thresh_OG)

        # store
        truncation_dict[x_OB] = (minM_OG, maxM_OG)

    return truncation_dict

# ------------------------------------------------
# Coefficients
# ------------------------------------------------

def Bm_matrix(x_OB, x_OG, beta):
    return scipy.stats.binom.pmf(x_OB, x_OG, beta)

def compute_coefficients(truncation_summary, truncation_OG, beta, name, thresh_OG=10**-6, tqdm_disable=True):

    # store marginal grids
    marginal_grids = {}

    # loop over observed counts
    for x_OB in tqdm.tqdm(truncation_summary['states'], disable=tqdm_disable):
        
        # get truncation
        minM_OG, maxM_OG = truncation_OG[x_OB]

        # construct arrays for broadcasting
        x_OB_arr = np.array([x_OB])[:, None]
        x_OG_arr = np.arange(minM_OG, maxM_OG + 1)[:, None]
        beta_arr = beta[None, :]
          
        # compute marginal grid
        marginal_grid = Bm_matrix(x_OB_arr, x_OG_arr, beta_arr)

        # store
        marginal_grids[x_OB] = marginal_grid

        # take mean over beta to get marginal coefficient array
        marginal_array = np.mean(marginal_grid, axis=1)

        # save
        np.save(
            f"./Temp/{name}/Coefficients/state-{x_OB}.npy",
            marginal_array
        )

    # loop over oberved count pairs
    for x1_OB, x2_OB in tqdm.tqdm(truncation_summary['state_pairs'], disable=tqdm_disable):

        # get marginal grids
        grid_x1_OB = marginal_grids[x1_OB]
        grid_x2_OB = marginal_grids[x2_OB]

        # compute outer product
        coeff_grid = grid_x1_OB @ grid_x2_OB.T

        # threshold
        coeff_grid[coeff_grid < thresh_OG] = 0.0

        # divide by sample size
        coeff_grid /= len(beta)

        # save
        np.save(
            f"./Temp/{name}/Coefficients/state-{x1_OB}-{x2_OB}.npy",
            coeff_grid
        )

# ------------------------------------------------
# Compute original truncation extent
# ------------------------------------------------

def compute_original_extent(truncation_OB, truncationM_OB, truncation_OG):

    # store per sample extent
    extent_dict = {}

    # for each sample
    for sample in truncation_OB.keys():

        # record min and max OG state extents
        min_x1_OG_ext, max_x1_OG_ext = np.inf, 0
        min_x2_OG_ext, max_x2_OG_ext = np.inf, 0

        # get OB truncation
        trunc_OB = truncation_OB[sample]

        # loop over OG truncation to get OG states used and update extent
        for x1_OB in range(trunc_OB['min_x1_OB'], trunc_OB['max_x1_OB'] + 1):
            min_x1_OG, max_x1_OG = truncation_OG[x1_OB]
            if min_x1_OG < min_x1_OG_ext:
                min_x1_OG_ext = min_x1_OG
            if max_x1_OG > max_x1_OG_ext:
                max_x1_OG_ext = max_x1_OG

        for x2_OB in range(trunc_OB['min_x2_OB'], trunc_OB['max_x2_OB'] + 1):
            min_x2_OG, max_x2_OG = truncation_OG[x2_OB]
            if min_x2_OG < min_x2_OG_ext:
                min_x2_OG_ext = min_x2_OG
            if max_x2_OG > max_x2_OG_ext:
                max_x2_OG_ext = max_x2_OG

        # get marginal OB truncation
        truncM_OB = truncationM_OB[sample]

        # repeat same process to update extent
        for x1_OB in range(truncM_OB['minM_x1_OB'], truncM_OB['maxM_x1_OB'] + 1):
            min_x1_OG, max_x1_OG = truncation_OG[x1_OB]
            if min_x1_OG < min_x1_OG_ext:
                min_x1_OG_ext = min_x1_OG
            if max_x1_OG > max_x1_OG_ext:
                max_x1_OG_ext = max_x1_OG

        for x2_OB in range(truncM_OB['minM_x2_OB'], truncM_OB['maxM_x2_OB'] + 1):
            min_x2_OG, max_x2_OG = truncation_OG[x2_OB]
            if min_x2_OG < min_x2_OG_ext:
                min_x2_OG_ext = min_x2_OG
            if max_x2_OG > max_x2_OG_ext:
                max_x2_OG_ext = max_x2_OG

        # store extent for the sample
        extent_dict[sample] = {
            'min_x1_OG': min_x1_OG_ext,
            'max_x1_OG': max_x1_OG_ext,
            'min_x2_OG': min_x2_OG_ext,
            'max_x2_OG': max_x2_OG_ext
        }

    return extent_dict