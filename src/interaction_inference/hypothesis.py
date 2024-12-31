'''
Implements class to handle hypothesis analysis method.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import tqdm
import pandas as pd
from interaction_inference import bootstrap
from interaction_inference import optimization

# ------------------------------------------------
# Hypothesis class
# ------------------------------------------------

class Hypothesis():
    def __init__(self):
        '''Initialize analysis settings and result storage.'''

        # bootstrap settings
        self.resamples = None
        self.splits = 1
        self.thresh_OB = 10
        self.threshM_OB = 10

        # analysis method settings
        self.license_filename = None
        self.constraint_settings = None
        self.time_limit = 300
        self.silent = True
        self.K = 100
        self.print_solution = False
        self.print_truncation = False

        # results
        self.result_dict = None

    def optimization_hypothesis(self):
        pass

    def analyse_dataset(self, dataset):
        '''Analyse given dataset using method settings and store results.'''

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

            # optimize: hyp
            if self.license_file:
                solution = optimization.optimization_hyp_WLS(
                                self.licence_file,
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
                                dataset.threshM_OG
                            )

            else:
                solution = optimization.optimization_hyp(
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
                                dataset.threshM_OG
                            )

            # store result
            solution_dict[i] = {'status': solution['status'], 'time': solution['time']}

        # store results
        self.result_dict = solution_dict
