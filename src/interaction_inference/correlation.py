'''
Implements class to handle correlation analysis method.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import scipy
import tqdm
import pandas as pd

# ------------------------------------------------
# Hypothesis class
# ------------------------------------------------

class Correlation():
    def __init__(self, method="pearson"):
        '''Initialize analysis settings and result storage.'''

        # analysis method settings
        self.method = method

        # results
        self.result_dict = None

    def analyse_dataset(self, dataset):
        '''Analyse given dataset using method settings and store results.'''

        # dict to store results
        solution_dict = {}

        # loop over dataset
        for i in tqdm.tqdm(range(dataset.gene_pairs)):

            # select sample
            sample = list(dataset.count_dataset.loc[f'Gene-pair-{i}'])

            if self.method == "pearson":

                # select individual samples
                x1_samples = [x[0] for x in samples]
                x2_samples = [x[1] for x in samples]

                # test
                pearson = scipy.stats.pearsonr(x1_samples, x2_samples)

                # store result
                solution_dict[i] = {'pvalue': float(pearson.pvalue), 'statistic': float(pearson.statistic)}

            elif self.method == "spearman":

                # select individual samples
                x1_samples = [x[0] for x in samples]
                x2_samples = [x[1] for x in samples]

                # test
                spearman = scipy.stats.spearmanr(x1_samples, x2_samples)

                # store result
                solution_dict[i] = {'pvalue': float(spearman.pvalue), 'statistic': float(spearman.statistic)}

        # store results
        self.result_dict = solution_dict
