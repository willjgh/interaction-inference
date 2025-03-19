'''
Module to compute confidence intervals from data for use in optimization.

Given a sample of pairs of counts from a pair of genes, e.g. scRNA-seq data, 
use bootstrap resampling to compute confidence interval bounds on the joint
and marginal distributions of the sample.

Typical example:

# get sample e.g. from dataset
sample = count_dataset.loc['Gene-pair-10']

# run bootstrap
bounds = bootstrap(sample, splits=10)
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

# ------------------------------------------------
# Bootstrap f's
# ------------------------------------------------

def bootstrap_f(sample, beta, resamples=None, thresh_OB=10, threshM_OB=10, printing=False):

    # get sample size
    n = len(sample)

    # get bootstrap size: default to sample size
    if resamples is None:
        resamples = n

    # initialize random generator
    rng = np.random.default_rng()

    # convert string to tuple if neccessary (pandas reading csv to string)
    if type(sample[0]) == str:
        sample = [literal_eval(count_pair) for count_pair in sample]

    # compute maximum x1 and x2 values
    M, N = np.max(sample, axis=0)
    M, N = int(M), int(N)

    # map (x1, x2) pairs to integers: x2 + (N + 1) * x1
    integer_sample = np.array([x[1] + (N + 1)*x[0] for x in sample], dtype='uint32')

    # maxiumum of integer sample
    D = (M + 1)*(N + 1) - 1

    # setup f arrays
    fm1m2 = np.zeros((2, M + 1, N + 1))
    fm1 = np.zeros((2, M + 1))
    fm2 = np.zeros((2, N + 1))

    # loop over states
    for m1 in range(M + 1):
        for m2 in range(N + 1):

            # capture for cells with counts (m1, m2)
            beta_m = beta[(sample == np.array([m1, m2])).sum(axis=1) == 2]

            # if empty
            if beta_m.size == 0:
                
                # store [0, 1] bounds
                fm1m2[:, m1, m2] = [0.0, 1.0]

                # move to next pair
                continue

            # bootstrap resample
            boot = rng.choice(beta_m, size=(resamples, len(beta_m)))

            # estimate E[beta|(m1, m2)]
            estimates = boot.mean(axis=1)

            # quantile for confidence intervals
            interval = np.quantile(estimates, [0.025, 0.975], axis=0)

            # store
            fm1m2[:, m1, m2] = interval

    # marginals
    x1_sample = np.array([x[0] for x in sample])
    x2_sample = np.array([x[1] for x in sample])

    for m1 in range(M + 1):

        # capture for cells with count m1
        beta_m = beta[x1_sample == m1]

        # if empty
        if beta_m.size == 0:
            
            # store [0, 1] bounds
            fm1[:, m1] = [0.0, 1.0]

            # move to next pair
            continue

        # bootstrap resample
        boot = rng.choice(beta_m, size=(resamples, len(beta_m)))

        # estimate E[beta|m]
        estimates = boot.mean(axis=1)

        # quantile for confidence intervals
        interval = np.quantile(estimates, [0.025, 0.975], axis=0)

        # store
        fm1[:, m1] = interval

    for m2 in range(N + 1):

        # capture for cells with count m2
        beta_m = beta[x2_sample == m2]

        # if empty
        if beta_m.size == 0:
            
            # store [0, 1] bounds
            fm2[:, m2] = [0.0, 1.0]

            # move to next pair
            continue

        # bootstrap resample
        boot = rng.choice(beta_m, size=(resamples, len(beta_m)))

        # estimate E[beta|m]
        estimates = boot.mean(axis=1)

        # quantile for confidence intervals
        interval = np.quantile(estimates, [0.025, 0.975], axis=0)

        # store
        fm2[:, m2] = interval

    # count occurances per (x1, x2) in the in original sample
    sample_counts = np.bincount(integer_sample, minlength=D + 1).reshape(M + 1, N + 1)

    # sum over columns / rows to give counts per x1 / x2 state
    x1_sample_counts = sample_counts.sum(axis=1)
    x2_sample_counts = sample_counts.sum(axis=0)
    
    # set truncation bounds
    min_x1_OB, max_x1_OB, min_x2_OB, max_x2_OB = M, 0, N, 0
    minM_x1_OB, maxM_x1_OB = M, 0
    minM_x2_OB, maxM_x2_OB = N, 0

    # set flag for changes
    thresh_flag = False
    thresh_flag_x1 = False
    thresh_flag_x2 = False

    # replace CI's for states below threshold occurances by [0, 1] bounds
    for x1 in range(M + 1):
        for x2 in range(N + 1):
            # below: replace
            if sample_counts[x1, x2] < thresh_OB:
                fm1m2[:, x1, x2] = [0.0, 1.0]
            # above: update truncation
            else:
                # check if smaller than current min
                if x1 < min_x1_OB:
                    min_x1_OB = x1
                    thresh_flag = True
                if x2 < min_x2_OB:
                    min_x2_OB = x2
                    thresh_flag = True
                # check if larger than current max
                if x1 > max_x1_OB:
                    max_x1_OB = x1
                    thresh_flag = True
                if x2 > max_x2_OB:
                    max_x2_OB = x2
                    thresh_flag = True

    for x1 in range(M + 1):
        # below: replace
        if x1_sample_counts[x1] < threshM_OB:
            fm1[:, x1] = [0.0, 1.0]
        # above: update truncation
        else:
            # check if smaller than current min
            if x1 < minM_x1_OB:
                minM_x1_OB = x1
                thresh_flag_x1 = True
            # check if larger than current max
            if x1 > maxM_x1_OB:
                maxM_x1_OB = x1
                thresh_flag_x1 = True

    for x2 in range(N + 1):
        # below: replace
        if x2_sample_counts[x2] < threshM_OB:
            fm2[:, x2] = [0.0, 1.0]
        # above: update truncation
        else:
            # check if smaller than current min
            if x2 < minM_x2_OB:
                minM_x2_OB = x2
                thresh_flag_x2 = True
            # check if larger than current max
            if x2 > maxM_x2_OB:
                maxM_x2_OB = x2
                thresh_flag_x2 = True

    # if no states were above threshold: default to max range, report
    if not thresh_flag:
        min_x1_OB, max_x1_OB, min_x2_OB, max_x2_OB = 0, M, 0, N
    if not thresh_flag_x1:
        minM_x1_OB, maxM_x1_OB = 0, M
    if not thresh_flag_x2:
        minM_x2_OB, maxM_x2_OB = 0, N

    # printing
    if printing:
        print(f"Box truncation: [{min_x1_OB}, {max_x1_OB}] x [{min_x2_OB}, {max_x2_OB}]")
        print(f"Marginal x1 truncation: [{minM_x1_OB}, {maxM_x1_OB}]")
        print(f"Marginal x2 truncation: [{minM_x2_OB}, {maxM_x2_OB}]")

    # collect results
    truncation_OB = {
        'min_x1_OB': min_x1_OB,
        'max_x1_OB': max_x1_OB,
        'min_x2_OB': min_x2_OB,
        'max_x2_OB': max_x2_OB
    }
    truncationM_OB = {
        'minM_x1_OB': minM_x1_OB,
        'maxM_x1_OB': maxM_x1_OB,
        'minM_x2_OB': minM_x2_OB,
        'maxM_x2_OB': maxM_x2_OB
    }

    result_dict = {
        'fm1m2': fm1m2,
        'fm1': fm1,
        'fm2': fm2,
        'truncation_OB': truncation_OB,
        'truncationM_OB': truncationM_OB
    }

    return result_dict

# ------------------------------------------------
# Bootstrap moments
# ------------------------------------------------

def bootstrap_moments(sample, beta, resamples=None):
    '''
    Compute confidence intervals on the moments of a sample of count pairs.

    Compute confidence intervals for the moments: mean, variance, cross moments,
    etc of the sample using the percentile bootstrap.

    Args:
        sample: list of tuples (x1, x2) of integer counts per cell
        beta: capture efficiency vector
        resamples: integer number of bootstrap resamples to use

    Returns:
        A dictionary containing results

        Confidence intervals:

        'E_x1': CI bounds on E[X1]
        'E_x2': CI bounds on E[X2]
        'E_x1_x2': CI ounds on E[X1X2]

        Truncation information:

        'max_x1_OG', 'max_x2_OG': marginal truncation extent
    '''

    # get sample size
    n = len(sample)

    # get bootstrap size: default to sample size
    if resamples is None:
        resamples = n

    # initialize random generator
    rng = np.random.default_rng()

    # convert string to tuple if neccessary (pandas reading csv to string)
    if type(sample[0]) == str:
        sample = [literal_eval(count_pair) for count_pair in sample]

    # separate sample pairs
    x1_sample = [x[0] for x in sample]
    x2_sample = [x[1] for x in sample]

    # convert sample to n x 2 array
    sample = np.array([x1_sample, x2_sample]).T

    # bootstrap to resamples x n x 2 array
    boot = rng.choice(sample, size=(resamples, n))

    # mean over axis 1 to get E[X1], E[X2] for each resample
    means = np.mean(boot, axis=1)

    # product over axis 2 to get x1x2 counts
    prods = np.prod(boot, axis=2)

    # mean over axis 1 to get E[X1X2] for each resample
    prod_means = np.mean(prods, axis=1)

    # square to get x1**2 and x2**2 counts
    squares = boot**2

    # mean over axis 1 to get E[X1**2], E[X2**2] for each resample
    square_means = np.mean(squares, axis=1)
    
    # quantiles over resamples
    mean_bounds = np.quantile(means, [0.025, 0.975], axis=0)
    prod_mean_bounds = np.quantile(prod_means, [0.025, 0.975], axis=0)
    square_mean_bounds = np.quantile(square_means, [0.025, 0.975], axis=0)

    # moment bounds
    moments_OB = {
        'E_x1': mean_bounds[:, 0],
        'E_x2': mean_bounds[:, 1],
        'E_x1_x2': prod_mean_bounds,
        'E_x1_sq': square_mean_bounds[:, 0],
        'E_x2_sq': square_mean_bounds[:, 1]
    }

    # OG truncation information

    # compute maximum x1 and x2 values
    max_x1_OB = int(np.max(x1_sample))
    max_x2_OB = int(np.max(x2_sample))

    # mean capture efficiency
    E_beta = np.mean(beta)

    # scale by mean capture to get max OG values
    max_x1_OG = int(max_x1_OB / E_beta) + 1
    max_x2_OG = int(max_x2_OB / E_beta) + 1

    # moment OG truncation
    truncation_OG = {
        'min_x1_OG': 0,
        'max_x1_OG': max_x1_OG,
        'min_x2_OG': 0,
        'max_x2_OG': max_x2_OG
    }

    # collect information
    result_dict = {
        'moments_OB': moments_OB,
        'truncation_OG': truncation_OG
    }

    return result_dict

# ------------------------------------------------
# Bootstrap probabilities
# ------------------------------------------------

def bootstrap_probabilities(sample, resamples=None, splits=1, thresh_OB=10, threshM_OB=10, plot=False, printing=False):
    '''
    Compute confidence intervals on the distribution of a sample of count pairs.

    Compute confidence intervals for the joint and marginal probabilities of the 
    sample using the percentile bootstrap and settings specified. Compute a state
    space truncation using a given threshold on the number of samples per interval,
    replacing intervals on probabilities of states outside the truncation by [0, 1]
    to improve coverage.

    Args:
        sample: list of tuples (x1, x2) of integer counts per cell
        resamples: integer number of bootstrap resamples to use
        splits: integer number of times to 'split' resampling across
                multiple arrays to avoid memory issues
        thresh_OB: threshold on observation frequency of a state pair
                   for state space truncation
        threshM_OB: threshold on observation frequency on a state for
                    marginal state space truncation
        
        plot: toggle plotting of confidence intervals and estimates
        print: toggle printing of observed state space truncation

    Returns:
        A dictionary containing results

        Confidence intervals:
    
        'bounds': (2, _, _) numpy array of CI bounds on joint distribution
        'x1_bounds': (2, _) numpy array of CI bounds on marginal distribution (gene 1)
        'x2_bounds': (2, _) numpy array of CI bounds on marginal distribution (gene 2)

        Truncation information:

        truncation_OB:

        'min_x1_OB', 'max_x1_OB', 'min_x2_OB', 'max_x2_OB': joint truncation

        truncationM_OB:

        'minM_x1_OB', 'maxM_x1_OB': marginal truncation (gene 1)
        'minM_x2_OB', 'maxM_x2_OB': marginal truncation (gene 2)
    '''

    # get sample size
    n = len(sample)

    # get bootstrap size: default to sample size
    if resamples is None:
        resamples = n

    # initialize random generator
    rng = np.random.default_rng()

    # convert string to tuple if neccessary (pandas reading csv to string)
    if type(sample[0]) == str:
        sample = [literal_eval(count_pair) for count_pair in sample]

    # compute maximum x1 and x2 values
    M, N = np.max(sample, axis=0)
    M, N = int(M), int(N)

    # map (x1, x2) pairs to integers: x2 + (N + 1) * x1
    integer_sample = np.array([x[1] + (N + 1)*x[0] for x in sample], dtype='uint32')

    # maxiumum of integer sample
    D = (M + 1)*(N + 1) - 1

    # number of bootstrap samples per split (split to reduce memory usage)
    resamples_split = resamples // splits

    # setup count array
    counts = np.empty((resamples, M + 1, N + 1), dtype='uint32')

    # BS bootstrap samples: split into 'splits' number of BS_split x n arrays
    for split in range(splits):

        # BS_split bootstrap samples as BS_split x n array
        bootstrap_split = rng.choice(integer_sample, size=(resamples_split, n))

        # offset row i by (D + 1)i
        bootstrap_split += np.arange(resamples_split, dtype='uint32')[:, None]*(D + 1)

        # flatten, count occurances of each state and reshape, reversing map to give counts of each (x1, x2) pair
        counts_split = np.bincount(bootstrap_split.ravel(), minlength=resamples_split*(D + 1)).reshape(-1, M + 1, N + 1)

        # add to counts
        counts[(split * resamples_split):((split + 1) * resamples_split), :, :] = counts_split

    # sum over columns / rows to give counts (/n) of each x1 / x2 state
    x1_counts = counts.sum(axis=2)
    x2_counts = counts.sum(axis=1)

    # compute 2.5% and 97.5% quantiles for each p(x1, x2), p(x1) and p(x2)
    bounds = np.quantile(counts, [0.025, 0.975], axis=0)
    x1_bounds = np.quantile(x1_counts, [0.025, 0.975], axis=0)
    x2_bounds = np.quantile(x2_counts, [0.025, 0.975], axis=0)

    # scale to probability
    bounds = bounds / n
    x1_bounds = x1_bounds / n
    x2_bounds = x2_bounds / n

    # count occurances per (x1, x2) in the in original sample
    sample_counts = np.bincount(integer_sample, minlength=D + 1).reshape(M + 1, N + 1)

    # sum over columns / rows to give counts per x1 / x2 state
    x1_sample_counts = sample_counts.sum(axis=1)
    x2_sample_counts = sample_counts.sum(axis=0)

    # set truncation bounds
    min_x1_OB, max_x1_OB, min_x2_OB, max_x2_OB = M, 0, N, 0
    minM_x1_OB, maxM_x1_OB = M, 0
    minM_x2_OB, maxM_x2_OB = N, 0

    # set flag for changes
    thresh_flag = False
    thresh_flag_x1 = False
    thresh_flag_x2 = False

    # replace CI's for states below threshold occurances by [0, 1] bounds
    for x1 in range(M + 1):
        for x2 in range(N + 1):
            # below: replace
            if sample_counts[x1, x2] < thresh_OB:
                bounds[:, x1, x2] = [0.0, 1.0]
            # above: update truncation
            else:
                # check if smaller than current min
                if x1 < min_x1_OB:
                    min_x1_OB = x1
                    thresh_flag = True
                if x2 < min_x2_OB:
                    min_x2_OB = x2
                    thresh_flag = True
                # check if larger than current max
                if x1 > max_x1_OB:
                    max_x1_OB = x1
                    thresh_flag = True
                if x2 > max_x2_OB:
                    max_x2_OB = x2
                    thresh_flag = True

    for x1 in range(M + 1):
        # below: replace
        if x1_sample_counts[x1] < threshM_OB:
            x1_bounds[:, x1] = [0.0, 1.0]
        # above: update truncation
        else:
            # check if smaller than current min
            if x1 < minM_x1_OB:
                minM_x1_OB = x1
                thresh_flag_x1 = True
            # check if larger than current max
            if x1 > maxM_x1_OB:
                maxM_x1_OB = x1
                thresh_flag_x1 = True

    for x2 in range(N + 1):
        # below: replace
        if x2_sample_counts[x2] < threshM_OB:
            x2_bounds[:, x2] = [0.0, 1.0]
        # above: update truncation
        else:
            # check if smaller than current min
            if x2 < minM_x2_OB:
                minM_x2_OB = x2
                thresh_flag_x2 = True
            # check if larger than current max
            if x2 > maxM_x2_OB:
                maxM_x2_OB = x2
                thresh_flag_x2 = True

    # if no states were above threshold: default to max range, report
    if not thresh_flag:
        min_x1_OB, max_x1_OB, min_x2_OB, max_x2_OB = 0, M, 0, N
    if not thresh_flag_x1:
        minM_x1_OB, maxM_x1_OB = 0, M
    if not thresh_flag_x2:
        minM_x2_OB, maxM_x2_OB = 0, N

    # plotting
    if plot:
        fig, axs = plt.subplots(M + 1, N + 1, figsize=(10, 10))
        fig.tight_layout()
        for x1 in range(M + 1):
            for x2 in range(N + 1):
                # within truncation: green CI lines
                if (x1 >= min_x1_OB) and (x2 >= min_x2_OB) and (x1 <= max_x1_OB) and (x2 <= max_x2_OB):
                    color = "green"
                else:
                    color = "red"
                axs[x1, x2].hist(counts[:, x1, x2] / n)
                axs[x1, x2].set_title(f"p({x1}, {x2})")
                axs[x1, x2].axvline(bounds[0, x1, x2], color=color)
                axs[x1, x2].axvline(bounds[1, x1, x2], color=color)

        plt.suptitle("X1 X2 Confidence Intervals")
        plt.show()

        fig, axs = plt.subplots(1, M + 1, figsize=(10, 3))
        fig.tight_layout()
        for x1 in range(M + 1):
            # within truncation: green CI lines
            if (x1 >= minM_x1_OB) and (x1 <= maxM_x1_OB):
                color = "green"
            else:
                color = "red"
            axs[x1].hist(x1_counts[:, x1] / n)
            axs[x1].set_title(f"p({x1})")
            axs[x1].axvline(x1_bounds[0, x1], color=color)
            axs[x1].axvline(x1_bounds[1, x1], color=color)

        plt.suptitle("X1 Confidence Intervals")
        plt.show()

        fig, axs = plt.subplots(1, N + 1, figsize=(10, 3))
        fig.tight_layout()
        for x2 in range(N + 1):
            # within truncation: green CI lines
            if (x2 >= minM_x2_OB) and (x2 <= maxM_x2_OB):
                color = "green"
            else:
                color = "red"
            axs[x2].hist(x2_counts[:, x2] / n)
            axs[x2].set_title(f"p({x2})")
            axs[x2].axvline(x2_bounds[0, x2], color=color)
            axs[x2].axvline(x2_bounds[1, x2], color=color)

        plt.suptitle("X2 Confidence Intervals")
        plt.show()

    # printing
    if printing:
        print(f"Box truncation: [{min_x1_OB}, {max_x1_OB}] x [{min_x2_OB}, {max_x2_OB}]")
        print(f"Marginal x1 truncation: [{minM_x1_OB}, {maxM_x1_OB}]")
        print(f"Marginal x2 truncation: [{minM_x2_OB}, {maxM_x2_OB}]")

    # collect results
    truncation_OB = {
        'min_x1_OB': min_x1_OB,
        'max_x1_OB': max_x1_OB,
        'min_x2_OB': min_x2_OB,
        'max_x2_OB': max_x2_OB
    }
    truncationM_OB = {
        'minM_x1_OB': minM_x1_OB,
        'maxM_x1_OB': maxM_x1_OB,
        'minM_x2_OB': minM_x2_OB,
        'maxM_x2_OB': maxM_x2_OB
    }

    result_dict = {
        'bounds': bounds,
        'x1_bounds': x1_bounds,
        'x2_bounds': x2_bounds,
        'truncation_OB': truncation_OB,
        'truncationM_OB': truncationM_OB
    }

    return result_dict