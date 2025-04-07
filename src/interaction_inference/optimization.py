'''
Module implementing class to handle optimization inference method.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

from interaction_inference import truncation
from interaction_inference import constraints
import json
import tqdm
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import traceback
from time import time

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
    def __init__(self, dataset, constraints, license_file=None, time_limit=300, silent=True, K=100, tqdm_disable=False, print_solution=True):
        '''Initialize analysis settings and result storage.'''
        
        # store reference to dataset
        self.dataset = dataset

        # constraints by list of names
        self.constraints = constraints

        # analysis settings
        self.license_file = license_file
        self.time_limit = time_limit
        self.silent = silent
        self.K = K
        self.tqdm_disable = tqdm_disable
        self.print_solution = print_solution

        # analyse dataset
        self.analyse_dataset()


    def analyse_dataset(self):
        '''Analyse given dataset using method settings and store results.'''

        # dict to store results
        solution_dict = {}

        # compute overall OG extent for constraints used
        self.overall_extent_OG = truncation.compute_overall_extent(
            self.constraints,
            self.dataset.moment_extent_OG,
            self.dataset.prob_extent_OG
        )

        # loop over gene pairs in dataset
        for i in tqdm.tqdm(range(self.dataset.gene_pairs), disable=self.tqdm_disable):

            # optimize sample i
            try:
                solution_dict[i] = self.optimize(i)

            # if exception
            except Exception as e:

                # display exception and traceback
                print(f"Optimization failed: {e}\n")
                print(traceback.format_exc())

                # store default result
                solution_dict[i] = {
                    'status': None,
                    'time': 0.0
                }

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

                # create variables
                variables = constraints.add_variables(self, model, i)

                # add constraints
                constraints.add_constraints(self, model, variables, i)
                
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