'''
Module implementing class to handle datasets and related settings.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

from interaction_inference import bootstrap
from interaction_inference import truncation
import pandas as pd
import numpy as np
import tqdm

# ------------------------------------------------
# Dataset class
# ------------------------------------------------

class Dataset():
    def __init__(self, name):
        '''Initialise dataset settings'''

        # name
        self.name = name

        # dataset iteself
        self.count_dataset = None
        self.param_dataset = None

        # size
        self.cells = None
        self.gene_pairs = None

        # capture efficiency
        self.beta = None

        # bootstrap settings
        self.resamples = None
        self.splits = 1
        self.thresh_OB = 10
        self.threshM_OB = 10
        
        # truncation settings
        self.thresh_OG = 10**-6

        # truncation info
        self.truncation_OB = None
        self.truncationM_OB = None
        self.truncation_OG = None

        # fm bounds
        self.fm_OB = None

        # moment bounds
        self.moments_OB = None
        self.moment_extent_OG = None

        # probability bounds
        self.probs_OB = None
        self.prob_extent_OG = None

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

    def downsample(self, name, beta):
        '''
        Apply a beta capture efficiency to the dataset, returning a copy with
        binomially downsampled counts and corresponding beta stored.
        '''

        # fail if dataset already downsampled
        if not (self.beta == np.array([1.0 for j in range(self.cells)])).all():
            print("Dataset has already been downsampled")
            return None

        # fail if incomptible cell numbers
        if not (beta.shape[0] == self.cells):
            print("Incompatible cell numbers.")
            return None
        
        # initialize random generator
        rng = np.random.default_rng()

        # setup downsampled dataset
        downsampled_counts_df = pd.DataFrame(
            index=[f"Gene-pair-{i}" for i in range(self.gene_pairs)],
            columns=[f"Cell-{j}" for j in range(self.cells)]
        )

        # for each sample
        for i in range(self.gene_pairs):

            # extract counts
            sample = self.count_dataset.iloc[i]
            x1_sample = [x[0] for x in sample]
            x2_sample = [x[1] for x in sample]

            # downsample
            x1_sample_downsampled = rng.binomial(x1_sample, beta).tolist()
            x2_sample_downsampled = rng.binomial(x2_sample, beta).tolist()
            sample_downsampled = list(zip(x1_sample_downsampled, x2_sample_downsampled))
            
            # store counts
            downsampled_counts_df.iloc[i] = sample_downsampled

        # create new downsampled dataset object
        downsampled_dataset = Dataset(name)

        # store counts
        downsampled_dataset.count_dataset = downsampled_counts_df

        # store capture
        downsampled_dataset.beta = beta

        # copy over information
        downsampled_dataset.param_dataset = self.param_dataset
        downsampled_dataset.cells = self.cells
        downsampled_dataset.gene_pairs = self.gene_pairs

        return downsampled_dataset
    
    def compute_fm(self, tqdm_disable=True):
        '''
        For each sample in dataset compute bootstrap CI bounds on fm = E[beta|m]
        rates and compute observed truncation information, storing this in
        attributes of the dataset.
        '''

        # collect OB truncations
        truncation_dict = {}
        truncationM_dict = {}

        # collect CI bounds
        fm_dict = {}

        # loop over samples
        for i in tqdm.tqdm(range(self.gene_pairs), disable=tqdm_disable):

            # select sample
            sample = list(self.count_dataset.loc[f'Gene-pair-{i}'])

            # bootstrap fms
            fm_results = bootstrap.bootstrap_fm(
                sample,
                self.beta,
                self.resamples,
                self.thresh_OB,
                self.threshM_OB
            )

            # store OB truncation
            truncation_dict[f'sample-{i}'] = fm_results['truncation_OB']
            truncationM_dict[f'sample-{i}'] = fm_results['truncationM_OB']

            # store CI bounds
            fm_dict[f'sample-{i}'] = {
                'fm1m2': fm_results['fm1m2'],
                'fm1': fm_results['fm1'],
                'fm2': fm_results['fm2']
            }

        # store information
        self.truncation_OB = truncation_dict
        self.truncationM_OB = truncationM_dict
        self.fm_OB = fm_dict

    def bootstrap_joint_fm(self, tqdm_disable=True):
        '''
        For each sample in dataset compute bootstrap CI bounds on fm1m2 = E[beta|(m1, m2)]
        rates and compute observed truncation information, storing this in
        attributes of the dataset.
        '''

        # collect OB truncations
        truncation_dict = {}

        # collect CI bounds if no existing dict
        if not self.fm_OB:
            fm_dict = {f'sample-{i}': {} for i in range(self.gene_pairs)}

        # loop over samples
        for i in tqdm.tqdm(range(self.gene_pairs), disable=tqdm_disable):

            # select sample
            sample = list(self.count_dataset.loc[f'Gene-pair-{i}'])

            # bootstrap probabilities
            fm_results = bootstrap.bootstrap_joint_fm(
                sample,
                self.beta,
                self.resamples,
                self.thresh_OB
            )

            # store OB truncation
            truncation_dict[f'sample-{i}'] = fm_results['truncation_OB']

            # store bounds in new dict
            if not self.fm_OB:
                fm_dict[f'sample-{i}']['fm1m2'] = fm_results['fm1m2']

            # or add to existing bounds dict
            else:
                self.fm_OB[f'sample-{i}']['fm1m2'] = fm_results['fm1m2']

        # store information
        self.truncation_OB = truncation_dict
        if not self.fm_OB:
            self.fm_OB = fm_dict

    def bootstrap_marginal_fm(self, tqdm_disable=True):
        '''
        For each sample in dataset compute bootstrap CI bounds on fm = E[beta|m]
        rates and compute observed truncation information, storing this in
        attributes of the dataset.
        '''
        # collect OB truncations
        truncationM_dict = {}

        # collect CI bounds if no existing dict
        if not self.fm_OB:
            fm_dict = {f'sample-{i}': {} for i in range(self.gene_pairs)}

        # loop over samples
        for i in tqdm.tqdm(range(self.gene_pairs), disable=tqdm_disable):

            # select sample
            sample = list(self.count_dataset.loc[f'Gene-pair-{i}'])

            # split into x1 and x2 samples
            x1_sample = [x[0] for x in sample]
            x2_sample = [x[1] for x in sample]

            # bootstrap marginal probabilities for x1
            fm_results_x1 = bootstrap.bootstrap_marginal_fm(
                x1_sample,
                self.beta,
                self.resamples,
                self.threshM_OB
            )

            # bootstrap marginal probabilities for x2
            fm_results_x2 = bootstrap.bootstrap_marginal_fm(
                x2_sample,
                self.beta,
                self.resamples,
                self.threshM_OB
            )

            # combine marginal truncations
            truncationM_OB = {
                'minM_x1_OB': fm_results_x1['truncationM_OB']['minM_x_OB'],
                'maxM_x1_OB': fm_results_x1['truncationM_OB']['maxM_x_OB'],
                'minM_x2_OB': fm_results_x2['truncationM_OB']['minM_x_OB'],
                'maxM_x2_OB': fm_results_x2['truncationM_OB']['maxM_x_OB']
            }

            # store OB truncation
            truncationM_dict[f'sample-{i}'] = truncationM_OB

            # store bounds in new dict
            if not self.fm_OB:
                fm_dict[f'sample-{i}']['fm1'] = fm_results_x1['fm']
                fm_dict[f'sample-{i}']['fm2'] = fm_results_x2['fm']

            # or add to existing bounds dict
            else:
                self.fm_OB[f'sample-{i}']['fm1'] = fm_results_x1['fm']
                self.fm_OB[f'sample-{i}']['fm2'] = fm_results_x2['fm']

        # store information
        self.truncationM_OB = truncationM_dict
        if not self.fm_OB:
            self.fm_OB = fm_dict
 
    def compute_moments(self, tqdm_disable=True):
        '''
        For each sample in dataset compute bootstrap CI bounds on moments
        and compute original truncation information, storing this in attributes
        of the dataset.
        '''

        # collect moment bounds
        moment_dict = {}

        # collect moment extent
        extent_dict = {}

        # loop over samples
        for i in tqdm.tqdm(range(self.gene_pairs), disable=tqdm_disable):

            # select sample
            sample = list(self.count_dataset.loc[f'Gene-pair-{i}'])

            # bootstrap moments
            moment_results = bootstrap.bootstrap_moments(
                sample,
                self.beta,
                self.resamples
            )

            # store moments
            moment_dict[f'sample-{i}'] = moment_results['moments_OB']

            # store extent
            extent_dict[f'sample-{i}'] = moment_results['truncation_OG']

        # store information
        self.moments_OB = moment_dict
        self.moment_extent_OG = extent_dict

    def bootstrap_probabilities(self, tqdm_disable=True):
        '''
        For each sample in dataset compute bootstrap CI bounds on joint and
        marginal probabilities as well as observed truncation information,
        storing this in attributes of the dataset.
        '''

        # collect OB truncations
        truncation_dict = {}
        truncationM_dict = {}

        # collect CI bounds
        probs_dict = {}

        # loop over samples
        for i in tqdm.tqdm(range(self.gene_pairs), disable=tqdm_disable):

            # select sample
            sample = list(self.count_dataset.loc[f'Gene-pair-{i}'])

            # bootstrap probabilities
            prob_results = bootstrap.bootstrap_probabilities(
                sample,
                self.resamples,
                self.splits,
                self.thresh_OB,
                self.threshM_OB
            )

            # store OB truncation
            truncation_dict[f'sample-{i}'] = prob_results['truncation_OB']
            truncationM_dict[f'sample-{i}'] = prob_results['truncationM_OB']

            # store CI bounds
            probs_dict[f'sample-{i}'] = {
                'bounds': prob_results['bounds'],
                'x1_bounds': prob_results['x1_bounds'],
                'x2_bounds': prob_results['x2_bounds']
            }

        # store information
        self.truncation_OB = truncation_dict
        self.truncationM_OB = truncationM_dict
        self.probs_OB = probs_dict

    def compute_probabilities(self, display=False, tqdm_disable=True):
        '''
        Additional setup needed for B-method probability constraints.
        '''

        # bootstrap probabilities
        self.bootstrap_probabilities(tqdm_disable=tqdm_disable)

        # display OB truncations
        if display:
            truncation.illustrate_truncation(self.truncation_OB, self.truncationM_OB)

        # summarise observed truncations
        truncation_summary = truncation.summarise_truncation(self.truncation_OB, self.truncationM_OB)

        # compute original truncation
        truncation_OG = truncation.original_truncation(truncation_summary, self.beta, self.thresh_OG, tqdm_disable=tqdm_disable)

        # compute and store B and Bm coefficients
        truncation.compute_coefficients(truncation_summary, truncation_OG, self.beta, self.name, self.thresh_OG, tqdm_disable=tqdm_disable)

        # compute original extent
        prob_extent_OG = truncation.compute_original_extent(self.truncation_OB, self.truncationM_OB, truncation_OG)

        # store information
        self.truncation_OG = truncation_OG
        self.prob_extent_OG = prob_extent_OG

    def bootstrap_joint_probabilities(self, tqdm_disable=True):
        '''
        For each sample in dataset compute bootstrap CI bounds on joint
        probabilities as well as observed truncation information,
        storing this in attributes of the dataset.
        '''

        # collect OB truncations
        truncation_dict = {}

        # collect CI bounds if no existing dict
        if not self.probs_OB:
            probs_dict = {f'sample-{i}': {} for i in range(self.gene_pairs)}

        # loop over samples
        for i in tqdm.tqdm(range(self.gene_pairs), disable=tqdm_disable):

            # select sample
            sample = list(self.count_dataset.loc[f'Gene-pair-{i}'])

            # bootstrap probabilities
            prob_results = bootstrap.bootstrap_joint_probabilities(
                sample,
                self.resamples,
                self.splits,
                self.thresh_OB
            )

            # store OB truncation
            truncation_dict[f'sample-{i}'] = prob_results['truncation_OB']

            # store bounds in new dict
            if not self.probs_OB:
                probs_dict[f'sample-{i}']['bounds'] = prob_results['bounds']

            # or add to existing bounds dict
            else:
                self.probs_OB[f'sample-{i}']['bounds'] = prob_results['bounds']

        # store information
        self.truncation_OB = truncation_dict
        if not self.probs_OB:
            self.probs_OB = probs_dict

    def bootstrap_marginal_probabilities(self, tqdm_disable=True):
        '''
        For each sample in dataset compute bootstrap CI bounds on marginal
        probabilities as well as observed truncation information,
        storing this in attributes of the dataset.
        '''

        # collect OB truncations
        truncationM_dict = {}

        # collect CI bounds if no existing dict
        if not self.probs_OB:
            probs_dict = {f'sample-{i}': {} for i in range(self.gene_pairs)}

        # loop over samples
        for i in tqdm.tqdm(range(self.gene_pairs), disable=tqdm_disable):

            # select sample
            sample = list(self.count_dataset.loc[f'Gene-pair-{i}'])

            # split into x1 and x2 samples
            x1_sample = [x[0] for x in sample]
            x2_sample = [x[1] for x in sample]

            # bootstrap marginal probabilities for x1
            prob_results_x1 = bootstrap.bootstrap_marginal_probabilities(
                x1_sample,
                self.resamples,
                self.splits,
                self.threshM_OB
            )

            # bootstrap marginal probabilities for x2
            prob_results_x2 = bootstrap.bootstrap_marginal_probabilities(
                x2_sample,
                self.resamples,
                self.splits,
                self.threshM_OB
            )

            # combine marginal truncations
            truncationM_OB = {
                'minM_x1_OB': prob_results_x1['truncationM_OB']['minM_x_OB'],
                'maxM_x1_OB': prob_results_x1['truncationM_OB']['maxM_x_OB'],
                'minM_x2_OB': prob_results_x2['truncationM_OB']['minM_x_OB'],
                'maxM_x2_OB': prob_results_x2['truncationM_OB']['maxM_x_OB']
            }

            # store OB truncation
            truncationM_dict[f'sample-{i}'] = truncationM_OB

            # store bounds in new dict
            if not self.probs_OB:
                probs_dict[f'sample-{i}']['x1_bounds'] = prob_results_x1['bounds']
                probs_dict[f'sample-{i}']['x2_bounds'] = prob_results_x2['bounds']

            # or add to existing bounds dict
            else:
                self.probs_OB[f'sample-{i}']['x1_bounds'] = prob_results_x1['bounds']
                self.probs_OB[f'sample-{i}']['x2_bounds'] = prob_results_x2['bounds']

        # store information
        self.truncationM_OB = truncationM_dict
        if not self.probs_OB:
            self.probs_OB = probs_dict