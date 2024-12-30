'''
Module implementing class to contain dataset settings.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import dataclass
import pandas as pd
import numpy as np

# ------------------------------------------------
# Dataset settings class
# ------------------------------------------------

class Dataset():

    def __init__(self):

        # dataset iteself
        self.count_dataset = None
        self.param_dataset = None

        # size
        self.cells = None
        self.gene_pairs = None

        # capture efficiency
        self.beta = None

        # simulation settings
        self.interaction = None
        self.conditional = None
        self.sig = None

        # truncation settings
        self.thresh_OG = 10**-6
        self.threshM_OG = 10**-6
        self.truncations = {}
        self.truncationsM = {}

    def load_dataset(self, beta, count_dataset_filename, param_dataset_filename=None):
        '''Load dataset from csv files: paramter and count data'''

        # load datasets
        self.count_dataset = pd.read_csv(count_dataset_filename, index_col=0)
        if param_dataset_filename:
            self.param_dataset = pd.read_csv(param_dataset_filename, index_col=0)

        # store shape and capture efficiency details
        self.gene_pairs, self.cells = self.count_dataset.shape
        self.beta = beta

    def store_dataset(self, count_dataset_filename, param_dataset_filename=None):
        '''Store dataset as csv files: parameter and count data'''

        # store datasets
        self.count_dataset.to_csv(count_dataset_filename)
        if param_dataset_filename:
            self.param_dataset.to_csv(param_dataset_filename)
        

    def simulate_dataset(self, cells, gene_pairs, beta, interaction, conditional, sig):
        # call function from module
        pass

    def pre_compute_truncation(self, size):
        # call compute truncation
        pass


@dataclass
class DatasetSettings:
    '''Class to store dataset and associated settings.'''
    dataset: pd.DataFrame = None
    dataset_filename: str = None
    cells: int = 1000
    gene_pairs: int = 100
    beta: np.array # could be Any to allow for float or array
    interaction: bool = True
    conditional: bool = False
    sig: float = 0.5
    thresh_OG: float = 10**-6
    threshM_OG: float = 10**-6
    truncations: dict = {}
    trunactionsM: dict = {}
    truncations_filename: str = None
    truncationsM_filename: str = None

# ------------------------------------------------
# Analysis settings class
# ------------------------------------------------

@dataclass
class AnalysisSettings:
    '''Class to store analysis settings and results.'''
    resamples: int
    splits: int
    thresh_OB: int
    threshM_OB: int
    license_filename: str
    constraint_settings: dict
    time_limit: int = 300
    silent: bool = True
    K: int = 100
    method: str
    print_solution: bool = False
    print_truncation: bool = False
    MIPGap: float = 0.05
    BestBdThresh: float = 0.0001
    result_dict: dict
    result_filename: str
