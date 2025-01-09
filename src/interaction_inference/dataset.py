'''
Module implementing class to handle datasets and related settings.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

from interaction_inference import simulation
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt

# ------------------------------------------------
# Dataset class
# ------------------------------------------------

class Dataset():
    def __init__(self):
        '''Initialise dataset settings'''

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

    def load_dataset(self, count_dataset_filename, beta=None, param_dataset_filename=None):
        '''Load dataset from csv files: paramter and count data'''
        self.count_dataset = pd.read_csv(count_dataset_filename, index_col=0)
        if param_dataset_filename:
            # parameter dataset only available for simulated data
            self.param_dataset = pd.read_csv(param_dataset_filename, index_col=0)

        # store shape and capture efficiency details
        self.gene_pairs, self.cells = self.count_dataset.shape
        self.beta = beta

    def store_dataset(self, count_dataset_filename, param_dataset_filename=None):
        '''Store dataset as csv files: parameter and count data'''
        self.count_dataset.to_csv(count_dataset_filename)
        if param_dataset_filename:
            # parameter dataset only available for simulated data
            self.param_dataset.to_csv(param_dataset_filename)

    def simulate_dataset(self, beta, gene_pairs, cells, interaction_chance=0.5, conditional=True, sig=0.5):
        '''
        Produce dataset of gene pairs' simulated parameters and samples.

        Produce a dataset of pairs of genes where for each pair parameters of a
        birth-death regulation model are simulated, and then a sample of size
        'cells' is simulated from the model. Parameters are sampled from log-uniform 
        distributions informed by single cell RNA sequencing experiments to produce
        realistic data. The paramters and samples are returned in a dictionary of 2
        pandas dataframes.

        Args:
            beta: per cell capture efficiency vector / single value
            gene_pairs: number of gene pairs to simulate
            cells: number of samples to simulate per gene pair
            interaction_chance: float in [0, 1], chance the interation parameter
                                'k_reg' is sampled vs being set to 0 (no interaction)
            conditional: toggle if model parameters for each gene in the pair are
                        sampled independently (False) or conditionally (True)
            sig: standard deviation about common value for parameters of each gene
                during conditional sampling 

        Returns:
            Nothing

            Sets attributes self.count_dataset and self.param_dataset to
            pandas dataframes of counts simulated per cell per gene-pair and
            model parameters simulated per gene-pair. Sets attributes with 
            simulation settings provided as arguments.
        '''

        # initialize random generator
        rng = np.random.default_rng()

        # dataframes
        params_df = pd.DataFrame(index=[f"Gene-pair-{i}" for i in range(gene_pairs)], columns=['k_tx_1', 'k_tx_2', 'k_deg_1', 'k_deg_2', 'k_reg'])
        counts_df = pd.DataFrame(index=[f"Gene-pair-{i}" for i in range(gene_pairs)], columns=[f"Cell-{j}" for j in range(cells)])

        # for each gene
        for i in tqdm.tqdm(range(gene_pairs)):

            # Select if interation or not
            u = rng.uniform()
            if u < interaction_chance:
                interaction = True
            else:
                interaction = False

            # Simulate reaction rate parameters 

            # conditional sampling
            if conditional:

                # sample log-mean capture efficiency
                log_mean = rng.uniform(-0.5, 1.5)

                # sample from normal distribution about the log-mean
                log_mean_1 = rng.normal(log_mean, sig)
                log_mean_2 = rng.normal(log_mean, sig)

                # sample degradation rates for each gene
                log_k_deg_1 = rng.uniform(-1, 0)
                log_k_deg_2 = rng.uniform(-1, 0)

                # compute transcription rates using log-mean and deg rate
                log_k_tx_beta_1 = log_mean_1 + log_k_deg_1
                log_k_tx_beta_2 = log_mean_2 + log_k_deg_2

                # sample interaction strength
                if interaction: log_k_reg = rng.uniform(-1, 1)

            # independent sampling
            else:

                # sample rates from log-uniform distribution for both genes
                log_k_tx_beta_1 = rng.uniform(-1, 1.5)
                log_k_tx_beta_2 = rng.uniform(-1, 1.5)
                log_k_deg_1 = rng.uniform(-1, 0)
                log_k_deg_2 = rng.uniform(-1, 0)
                if interaction: log_k_reg = rng.uniform(-2, 2)

            # exponentiate and scale
            k_tx_1 = (10 ** log_k_tx_beta_1) / np.mean(beta)
            k_tx_2 = (10 ** log_k_tx_beta_2) / np.mean(beta)
            k_deg_1 = 10 ** log_k_deg_1
            k_deg_2 = 10 ** log_k_deg_2
            if interaction:
                k_reg = 10 ** log_k_reg
            else:
                k_reg = 0

            # store parameters
            params_df.iloc[i] = [k_tx_1, k_tx_2, k_deg_1, k_deg_2, k_reg]

            params = {
                'k_tx_1': k_tx_1,
                'k_tx_2': k_tx_2,
                'k_deg_1': k_deg_1,
                'k_deg_2': k_deg_2,
                'k_reg': k_reg
            }

            # simulate sample from model
            sample = simulation.gillespie(params, cells, beta)

            # store counts
            counts_df.iloc[i] = sample['OB']

        # store datasets
        self.count_dataset = counts_df
        self.param_dataset = params_df

        # store simulation settings
        self.beta = beta
        self.gene_pairs = gene_pairs
        self.cells = cells
        self.interaction = interaction
        self.conditional = conditional
        if conditional:
            self.sig = sig