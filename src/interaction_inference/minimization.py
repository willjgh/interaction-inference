'''
Module implementing class to handle minimization inference method.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import tqdm
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from interaction_inference import bootstrap
from interaction_inference import truncation

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

        # loop over gene pairs in dataset
        for i in tqdm.tqdm(range(dataset.gene_pairs)):

            # select sample
            sample = list(dataset.count_dataset.loc[f'Gene-pair-{i}'])

            # bootstrap
            bounds = bootstrap.bootstrap(sample, self)

            # optimize
            solution = self.optimize(bounds, dataset)

            # store result
            solution_dict[i] = solution

        # add as attribute
        self.result_dict = solution_dict

    def optimize(self, bounds, dataset):
        '''
        Optimize lower bound on interaction parameter of a birth-death model.

        Optimize lower bound on the interaction parameter k_reg using constraints
        from the birth-death model structure and confidence intervals on the 
        distribution of observed data. Solve to determine if the lower bound is
        zero: and so the data is consistent with a model of no interaction, or 
        non-zero: suggesting interaction is present.

        Args:
            bounds: dictionary of confidence interval bounds computed by the 
                    bootstrap on the distribution of a sample from the dataset
            dataset: instance of Dataset class containing capture efficiency
                     and truncation settings used to construct constraints

        Returns:
            A dictionary containing results

            'bound': lower bound on k_reg
            'status': GUROBI optimization status, typically 'OPTIMAL', 'USER_OBJ_LIMIT',
                    'INFEASIBLE' or 'TIME_LIMIT' (see status codes below / in docs)
            'time': optimization solver runtime
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
            with gp.Model('birth-death-interaction-minimization', env=env) as md:

                # optimization settings
                md.Params.TimeLimit = self.time_limit # time limit
                md.Params.MIPGap = self.MIPGap # optimal gap for termination
                md.Params.Presolve = 2 # aggressive presolve
                md.Params.MIPFocus = 3 # focus on lower bound of objective: allows early termination
                md.Params.BestBdStop = self.BestBdThresh # threshold on lower bound for termination

                # State space truncations

                # observed truncations: computed during bootstrap
                min_x1_OB = bounds['min_x1_OB']
                max_x1_OB = bounds['max_x1_OB']
                min_x2_OB = bounds['min_x2_OB']
                max_x2_OB = bounds['max_x2_OB']

                # original truncations: find largest original states needed (to define variables)
                overall_min_x1_OG, overall_max_x1_OG, overall_min_x2_OG, overall_max_x2_OG = np.inf, 0, np.inf, 0

                # for each pair of observed states used
                for x1_OB in range(min_x1_OB, max_x1_OB + 1):
                    for x2_OB in range(min_x2_OB, max_x2_OB + 1):

                        try:
                            # lookup original truncation
                            min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = dataset.truncations[f'({x1_OB}, {x2_OB})']

                        except KeyError:
                            # compute if not available
                            min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = truncation.compute_state_trunc(x1_OB, x2_OB, dataset.beta, dataset.thresh_OG)

                            # store
                            dataset.truncations[f'({x1_OB}, {x2_OB})'] = (min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG)

                            # store symmetry
                            dataset.truncations[f'({x2_OB}, {x1_OB})'] = (min_x2_OG, max_x2_OG, min_x1_OG, max_x1_OG)

                        # if larger than current maximum states: update
                        if max_x1_OG > overall_max_x1_OG:
                            overall_max_x1_OG = max_x1_OG
                        if max_x2_OG > overall_max_x2_OG:
                            overall_max_x2_OG = max_x2_OG

                        # if smaller than current minimum states: update
                        if min_x1_OG < overall_min_x1_OG:
                            overall_min_x1_OG = min_x2_OG
                        if min_x2_OG < overall_min_x2_OG:
                            overall_min_x2_OG = min_x2_OG

                if self.print_truncation:
                    print(f"Observed counts: [{min_x1_OB}, {max_x1_OB}] x [{min_x2_OB}, {max_x2_OB}]")
                    print(f"Original counts: [{overall_min_x1_OG}, {overall_max_x1_OG}] x [{overall_min_x2_OG}, {overall_max_x2_OG}]")

                # variables

                # stationary distribution: original counts (size = largest truncation)
                p = md.addMVar(shape=(overall_max_x1_OG + 1, overall_max_x2_OG + 1), vtype=GRB.CONTINUOUS, name="p", lb=0, ub=1)

                # reaction rate constants
                rate_names = ['k_tx_1', 'k_tx_2', 'k_deg_1', 'k_deg_2', 'k_reg']
                rates = md.addVars(rate_names, vtype=GRB.CONTINUOUS, lb=0, ub=self.K, name=rate_names)

                # constraints

                # fix k_deg_2 = 1 for identifiability
                md.addConstr(rates['k_deg_2'] == 1)

                # distribution
                md.addConstr(p.sum() <= 1, name="Distribution")

                # stationary distribution bounds: for each observed count pair
                for x1_OB in range(min_x1_OB, max_x1_OB + 1):
                    for x2_OB in range(min_x2_OB, max_x2_OB + 1):
                        
                        # original truncation: lookup from pre-computed dict
                        min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = dataset.truncations[f'({x1_OB}, {x2_OB})']
                        
                        # sum over truncation range (INCLUSIVE): drop terms with coefficients < thresh
                        sum_expr = gp.quicksum([
                            truncation.B_coeff(x1_OB, x2_OB, x1_OG, x2_OG, dataset.beta) * p[x1_OG, x2_OG]
                            for x1_OG in range(min_x1_OG, max_x1_OG + 1)
                            for x2_OG in range(min_x2_OG, max_x2_OG + 1)
                            if truncation.B_coeff(x1_OB, x2_OB, x1_OG, x2_OG, dataset.beta) >= dataset.thresh_OG
                        ])
                        
                        md.addConstr(sum_expr >= bounds['joint'][0, x1_OB, x2_OB], name=f"B lb {x1_OB}, {x2_OB}")
                        md.addConstr(sum_expr <= bounds['joint'][1, x1_OB, x2_OB], name=f"B ub {x1_OB}, {x2_OB}")
                
                # stationary Qp=0 equations for all relevant variables
                for x1_OG in range(overall_max_x1_OG):
                    for x2_OG in range(overall_max_x2_OG):

                        # remove terms when x's = 0 as not present in equation
                        if x1_OG == 0:
                            x1_zero = 0
                        else:
                            x1_zero = 1
                        if x2_OG == 0:
                            x2_zero = 0
                        else:
                            x2_zero = 1

                        md.addConstr(
                            rates['k_tx_1'] * x1_zero * p[x1_OG - 1, x2_OG] + \
                            rates['k_tx_2'] * x2_zero * p[x1_OG, x2_OG - 1] + \
                            rates['k_deg_1'] * (x1_OG + 1) * p[x1_OG + 1, x2_OG] + \
                            rates['k_deg_2'] * (x2_OG + 1) * p[x1_OG, x2_OG + 1] + \
                            rates['k_reg'] * (x1_OG + 1) * (x2_OG + 1) * p[x1_OG + 1, x2_OG + 1] - \
                            (rates['k_tx_1'] + rates['k_tx_2'] + \
                            rates['k_deg_1'] * x1_OG + rates['k_deg_2'] * x2_OG + \
                            rates['k_reg'] * x1_OG * x2_OG) * p[x1_OG, x2_OG] == 0,
                            name=f"Equation {x1_OG}, {x2_OG}"
                            )

                # status of optimization
                status_codes = {1: 'LOADED',
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
                                15: 'USER_OBJ_LIMIT'}

                # solution dict
                solution = {}

                # set objective: minimize interaction parameter
                md.setObjective(rates['k_reg'], GRB.MINIMIZE)

                # optimize
                try:
                    md.optimize()
                    min_val = md.ObjVal
                except:
                    min_val = None

                # store lower bound
                solution['bound'] = min_val 

                # store status
                solution['status'] = status_codes[md.status]

                # store optimization time
                solution['time'] = md.Runtime
        
        # print
        if self.print_solution:
            print(f"k_reg lower bound: {solution['bound']}")
            print(f"Optimization status: {solution['status']}")
            print(f"Runtime: {solution['time']}")

        return solution