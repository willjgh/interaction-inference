'''
Module implementing class to handle optimization inference method.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

from interaction_inference import truncation
import json
import tqdm
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# ------------------------------------------------
# Constants
# ------------------------------------------------

status_codes = {
    1: 'LOADED',
    2: 'OPTIMAL',
    3: 'INFEASIBLE',
    4: 'INF_OR_UNBD',
    5: 'UNBOUNDED',
    6: 'CUTOFF',
    7: 'ITERATION_LIMIT',
    8: 'NODE_LIMIT',
    9: 'TIME_LIMIT',
    10: 'SOLUTION_LIMIT',
    11: 'INTERRUPTED',
    12: 'NUMERIC',
    13: 'SUBOPTIMAL',
    14: 'INPROGRESS',
    15: 'USER_OBJ_LIMIT'
}

# ------------------------------------------------
# Hypothesis class
# ------------------------------------------------

class Optimization():
    def __init__(self, dataset, constraints, tqdm_disable=False, printing=True):
        '''Initialize analysis settings and result storage.'''
        
        # store reference to dataset
        self.dataset = dataset

        # constraints by list of names
        self.constraints = constraints

        # analysis settings
        self.license_file = None
        self.time_limit = 300
        self.silent = True
        self.K = 100
        self.print_solution = False

        # analyse dataset
        self.analyse_dataset(tqdm_disable=tqdm_disable, printing=printing)


    def analyse_dataset(self, tqdm_disable=False, printing=True):
        '''Analyse given dataset using method settings and store results.'''

        # dict to store results
        solution_dict = {}

        # compute overall OG extent for constraints used
        self.overall_extent = truncation.compute_overall_extent(
            self.constraints,
            self.dataset.moment_extent_OG,
            self.dataset.prob_extent_OG
        )

        # loop over gene pairs in dataset
        for i in tqdm.tqdm(range(self.dataset.gene_pairs), tqdm_disable=tqdm_disable):

            # optimize sample i
            solution_dict[i] = self.optimize(i)

        # store as attribute
        self.result_dict = solution_dict


    def optimize(self, i):
        '''
        Construct constraints of feasibility test for sample i and optimize.
        '''
        
        # if provided load WLS license credentials
        if self.license_file:
            environment_parameters = json.load(open(self.license_file))
        # otherwise use default environment (e.g Named User license)
        else:
            environment_parameters = {}
 
        # silence output
        if self.silent:
            environment_parameters['OutputFlag'] = 0
        
        # environment context
        with gp.Env(params=environment_parameters) as env:

            # model context
            with gp.Model('model', env=env) as model:

                # set optimization parameters
                model.Params.TimeLimit = self.time_limit
                K = 100

                # create variables

                # marginal stationary distributions
                p1 = model.addMVar(shape=(self.overall_extent[f'sample-{i}']['max_x1_OG'] + 1), vtype=GRB.CONTINUOUS, name="p1", lb=0, ub=1)
                p2 = model.addMVar(shape=(self.overall_extent[f'sample-{i}']['max_x2_OG'] + 1), vtype=GRB.CONTINUOUS, name="p2", lb=0, ub=1)

                # reaction rate constants
                rate_names = ['k_tx_1', 'k_tx_2', 'k_deg_1', 'k_deg_2']
                rates = model.addVars(rate_names, vtype=GRB.CONTINUOUS, lb=0, ub=K, name=rate_names)

                # collect variables
                variables = {
                    'p1': p1,
                    'p2': p2,
                    'k_tx_1': rates['k_tx_1'],
                    'k_tx_2': rates['k_tx_2'],
                    'k_deg_1': rates['k_deg_1'],
                    'k_deg_2': rates['k_deg_2']
                }

                # add constraints
                '''
                TODO
                '''
                
                # optimize: test feasibility
                model.setObjective(0, GRB.MINIMIZE)
                try:
                    model.optimize()
                except gp.GurobiError:
                    print("GurobiError")

                # collect solution information
                solution = {
                    'status': status_codes[model.status],
                    'time': model.Runtime
                }

        # print
        if self.print_solution:
            print(f"Optimization status: {solution['status']}")
            print(f"Runtime: {solution['time']}")

        return solution