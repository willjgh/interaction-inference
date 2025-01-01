'''
Implements class to handle minmization analysis method.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import tqdm
import pandas as pd
from interaction_inference import bootstrap
from interaction_inference import optimization

# ------------------------------------------------
# Minimization class
# ------------------------------------------------

class Minimization():
    def __init__(self):
        '''Initialize analysis settings and result storage.'''

        # bootstrap settings
        self.resamples = None
        self.splits = 1
        self.thresh_OB = 10
        self.threshM_OB = 10

        # analysis method settings
        self.method = "min"
        self.license_file = None
        self.settings = None
        self.time_limit = 300
        self.silent = True
        self.K = 100
        self.print_solution = False
        self.print_truncation = False
        self.MIPGap = 0.05
        self.BestBdThresh = 0.0001

        # dataset used
        self.dataset = None

        # results
        self.result_dict = None

    def analyse_dataset(self, dataset):
        '''Analyse given dataset using method settings and store results.'''

        # store reference to dataset used
        self.dataset = dataset

        # dict to store results
        solution_dict = {}

        # loop over dataset
        for i in tqdm.tqdm(range(dataset.gene_pairs)):

            # select sample
            sample = list(dataset.count_dataset.loc[f'Gene-pair-{i}'])

            # bootstrap
            bounds = bootstrap.bootstrap(
                        sample,
                        self.resamples,
                        self.splits,
                        dataset.beta,
                        self.thresh_OB,
                        self.threshM_OB,
                        plot=False,
                        printing=False
                    )

            # optimize: min
            if self.license_file:
                solution = optimization.optimization_min_WLS(
                                self.license_file,
                                bounds,
                                dataset.beta,
                                self.settings,
                                self.time_limit,
                                self.silent,
                                self.K,
                                self.print_solution,
                                self.print_truncation,
                                dataset.truncations,
                                dataset.truncationsM,
                                dataset.thresh_OG,
                                dataset.threshM_OG,
                                self.MIPGap,
                                self.BestBdThresh
                            )

            else:
                solution = optimization.optimization_min(
                                bounds,
                                dataset.beta,
                                self.settings,
                                self.time_limit,
                                self.silent,
                                self.K,
                                self.print_solution,
                                self.print_truncation,
                                dataset.truncations,
                                dataset.truncationsM,
                                dataset.thresh_OG,
                                dataset.threshM_OG,
                                self.MIPGap,
                                self.BestBdThresh
                            )

            # store result
            solution_dict[i] = {'bound': solution['bound'], 'status': solution['status'], 'time': solution['time']}

        # store results
        self.result_dict = solution_dict
