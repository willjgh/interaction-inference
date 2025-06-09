'''
Module to compute state space truncations for use in optimization constraints.

Given a coefficient threshold and per cell capture efficiency vector compute
a dictionary of truncations. Should be computed once per dataset and passed as
a dictionary to optimization or stored as a json file for later use.

Typical example:

# set capture efficiency and threshold
beta = 0.5
thresh = 10**-6

# compute 20 x 20 grid of joint truncations
truncations = compute_truncation(20, beta, thresh)

# compute 20 marginal truncations
truncationsM = compute_truncationM(20, beta, thresh)
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import numpy as np
import scipy
import json
import tqdm

# ------------------------------------------------
# Functions
# ------------------------------------------------

def B_coeff(x1_OB, x2_OB, x1_OG, x2_OG, beta):
    '''Compute (1 / n) sum j = 1 to n of P(X1_OB, X2_OB | X1_OG, X2_OG, beta_j): product of binomial pmfs.'''
    
    return scipy.stats.binom.pmf(x1_OB, x1_OG, beta) * scipy.stats.binom.pmf(x2_OB, x2_OG, beta)

def BM_coeff(x_OB, x_OG, beta):
    '''Compute (1 / n) sum j = 1 to n of P(X_OB | X_OG, Beta_j): binomial pmfs.'''

    return scipy.stats.binom.pmf(x_OB, x_OG, beta)

def compute_state_trunc(x1_OB, x2_OB, beta, thresh_OG):
    '''
    Compute box truncation around states (x1_OG, x2_OG) which have
    B_coeff(x1_OB, x2_OB, x1_OG, x2_OG, beta) >= thresh_OG

    returns: min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG
    '''

    trunc_start = False
    trunc_end = False
    min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = np.inf, 0, np.inf, 0
    diag = 0
    while (not trunc_start) or (not trunc_end):

        # flag if at least one coeff > thresh in diagonal
        trunc_diag = False

        # diagonal from upper right to lower left
        x1_OG_diag = np.array([x1_OB + i for i in range(diag + 1)])
        x2_OG_diag = np.array([x2_OB + diag - i for i in range(diag + 1)])

        # compute coeffs
        coeffs = B_coeff(x1_OB, x2_OB, x1_OG_diag, x2_OG_diag, beta)

        # find where above threshold
        idxs = np.argwhere(coeffs > 10**-6).reshape(-1)

        # if any values above threshold
        if idxs.size > 0:

            # at least one coeff > thresh (overall)
            trunc_start = True

            # at least one coeff > thresh (in diag)
            trunc_diag = True

            # find states above threshold
            x1_states = x1_OG_diag[idxs]
            x2_states = x2_OG_diag[idxs]

            # find boundaries
            min_x1 = min(x1_states)
            min_x2 = min(x2_states)
            max_x1 = max(x1_states)
            max_x2 = max(x2_states)

            # update truncations
            if min_x1 < min_x1_OG:
                min_x1_OG = min_x1
            if min_x2 < min_x2_OG:
                min_x2_OG = min_x2
            if max_x1 > max_x1_OG:
                max_x1_OG = max_x1
            if max_x2 > max_x2_OG:
                max_x2_OG = max_x2

        # if NO coeff > thresh (in diag) AND at least one coeff > thresh (overall)
        if (not trunc_diag) and trunc_start:

            # end
            trunc_end = True

        # increment diagonal
        diag += 1

    # cast to int
    min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = int(min_x1_OG), int(max_x1_OG), int(min_x2_OG), int(max_x2_OG)

    return min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG

def compute_state_truncM(x_OB, beta, threshM_OG):
    '''
    Compute interval truncation of states x_OG which have
    B(x_OB, x_OG, beta) >= threshM_OG
    
    returns: minM_OG, maxM_OG
    '''

    # start at first non-zero coefficient
    x_OG = x_OB
    coeff = BM_coeff(x_OB, x_OG, beta)

    # if not above threshold: increment until above
    while coeff < threshM_OG:

        # increment
        x_OG += 1

        # compute coeff
        coeff = BM_coeff(x_OB, x_OG, beta)

    # store first state coeff >= thresh
    minM_OG = x_OG

    # increment until below threshold
    while coeff >= threshM_OG:

        # increment
        x_OG += 1

        # compute coeff
        coeff = BM_coeff(x_OB, x_OG, beta)

    # store last state with coeff >= thresh (INCLUSIVE BOUND)
    maxM_OG = x_OG - 1

    return minM_OG, maxM_OG

def compute_truncation(size, beta, thresh_OG):
    '''
    Compute dictionary of truncations for original state pairs

    size: grid size of observed pairs that truncations are computed for
    beta: capture efficiency vector
    thresh_OG: threshold for trunction
    '''
    # store in dictionary
    truncations = {}

    # for each pair of observed counts
    for x1_OB in tqdm.tqdm(range(size)):
        for x2_OB in range(x1_OB + 1):

            # compute truncation bounds
            min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = compute_state_trunc(x1_OB, x2_OB, beta, thresh_OG)

            # store
            truncations[f'({x1_OB}, {x2_OB})'] = (min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG)

            # store symmetric version
            truncations[f'({x2_OB}, {x1_OB})'] = (min_x2_OG, max_x2_OG, min_x1_OG, max_x1_OG)

    return truncations

def compute_truncationM(size, beta, threshM_OG):
    '''
    Compute dict of original truncations

    size: number of states that truncations are computed for
    beta: capture efficiency vector
    threshM_OG: threshold for trunction
    '''
    # store in dictionary
    truncations = {}

    # for each observed count
    for x_OB in tqdm.tqdm(range(max)):

        # compute truncation bounds
        minM_OG, maxM_OG = compute_state_truncM(x_OB, beta, threshM_OG)

        # store
        truncations[f'{x_OB}'] = (minM_OG, maxM_OG)

    return truncations
