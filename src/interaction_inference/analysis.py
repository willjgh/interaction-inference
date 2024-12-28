'''
Module to analyse datasets using optimization and correlation inference methods.
'''

import scipy
import tqdm
import pandas as pd
import numpy as np
from interaction_inference import bootstrap
from interaction_inference import optimization

# dataset analysis script as function
def dataset_analysis(rng, counts_df, beta, resamples=None, splits=None, thresh_OB=10, threshM_OB=10,
                     method, settings=None, truncations={}, truncationsM={}, license_file, K=100,
                     silent=True, print_solution=False, print_truncation=False, thresh_OG=10**-6, threshM_OG=10**-6,
                     time_limit=300, MIPGap=0.05, BestBdThresh=0.0001):

    # size of dataset
    gene_pairs, cells = counts_df.shape

    # default bootstrap size to sample size
    if resamples is None:
        resamples = cells

    # default bootstrap splits to resamples // 1000
    splits = resamples // 1000

    # dict to store results
    solution_dict = {}

    # loop over dataset
    for i in tqdm.tqdm(range(gene_pairs)):

        # select sample
        samples = list(counts_df.loc[f'Gene-pair-{i}'])

        if method == "hyp":

            # bootstrap
            bounds = bootstrap.bootstrap(rng, sample, resamples, splits, beta, thresh_OB, threshM_OB, plot=False, printing=False)

            # optimize: hyp
            if license_file:
                solution = optimization.optimization_hyp_WLS(licence_file, bounds, beta, settings, time_limit, silent,
                                                K, print_solution, print_truncation, truncations, truncationsM,
                                                thresh_OG, threshM_OG)

            else:
                solution = optimization.optimization_hyp(bounds, beta, settings, time_limit, silent,
                                            K, print_solution, print_truncation,
                                            truncations, truncationsM, thresh_OG, threshM_OG)

            # store result
            solution_dict[i] = {'status': solution['status'], 'time': solution['time']}

        elif method == "min":

            # bootstrap
            bounds = bootstrap.bootstrap(rng, sample, resamples, splits, beta, thresh_OB, threshM_OB, plot=False, printing=False)

            # optimize: min
            if license_file:
                solution = optimization.optimization_min_WLS(license_file, bounds, beta, settings, time_limit, silent,
                                                K, print_solution, print_truncation, truncations, truncationsM,
                                                thresh_OG, threshM_OG, MIPGap, BestBdThresh)

            else:
                solution = optimization.optimization_min_WLS(bounds, beta, settings, time_limit, silent,
                                                K, print_solution, print_truncation, truncations, truncationsM,
                                                thresh_OG, threshM_OG, MIPGap, BestBdThresh)

            # store result
            solution_dict[i] = {'bound': solution['k_reg'][0], 'status': solution['k_reg'][1], 'time': solution['time']}

        elif method == "pearson":

            # select individual samples
            x1_samples = [x[0] for x in samples]
            x2_samples = [x[1] for x in samples]

            # test
            pearson = scipy.stats.pearsonr(x1_samples, x2_samples)

            # store result
            solution_dict[i] = {'pvalue': float(pearson.pvalue), 'statistic': float(pearson.statistic)}

        elif method == "spearman":

            # select individual samples
            x1_samples = [x[0] for x in samples]
            x2_samples = [x[1] for x in samples]

            # test
            spearman = scipy.stats.spearmanr(x1_samples, x2_samples)

            # store result
            solution_dict[i] = {'pvalue': float(spearman.pvalue), 'statistic': float(spearman.statistic)}

    return solution_dict