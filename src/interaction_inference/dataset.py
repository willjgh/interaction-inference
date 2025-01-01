'''
Module implementing class to contain dataset settings.
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
# Dataset settings class
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

    def load_dataset(self, beta, count_dataset_filename, param_dataset_filename=None):
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

    def pre_compute_truncation(self, size):
        # call compute truncation
        pass

    def store_truncation(self):
        # store as json
        pass

    def load_truncation(self):
        # load json
        pass

    def simulate_dataset(self, beta, gene_pairs, cells, interaction=True, conditional=False, sig=0.5, plot=False):
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
            interaction: toggle if the interation parameter 'k_reg' is sampled (True)
                        or set to 0 (False)
            conditional: toggle if model parameters for each gene in the pair are
                        sampled independently (False) or conditionally (True)
            sig: standard deviation about common value for parameters of each gene
                during conditional sampling 
            plot: toggle plotting a scatter plot of the simulated parameters

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

        # plot
        if plot:
            plt.scatter(np.log10(params_df['k_tx_1'].astype(np.float64)) - np.log10(params_df['k_deg_1'].astype(np.float64)), np.log10(params_df['k_tx_2'].astype(np.float64)) - np.log10(params_df['k_deg_2'].astype(np.float64)))
            plt.xlabel("log(k_tx_1 / k_deg_1)")
            plt.ylabel("log(k_tx_2 / k_deg_2)")
            plt.title("Scatter plot of gene-pair parameters")
            plt.show()

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