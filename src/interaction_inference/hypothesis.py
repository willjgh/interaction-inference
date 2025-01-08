'''
Implements class to handle hypothesis analysis method.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import tqdm
import pandas as pd
from interaction_inference import bootstrap
from interaction_inference import truncation

# ------------------------------------------------
# Hypothesis class
# ------------------------------------------------

class Hypothesis():
    def __init__(self):
        '''Initialize analysis settings and result storage.'''

        # bootstrap settings
        self.method = "hyp"
        self.resamples = None
        self.splits = 1
        self.thresh_OB = 10
        self.threshM_OB = 10

        # analysis method settings
        self.license_file = None
        self.settings = None
        self.time_limit = 300
        self.silent = True
        self.K = 100
        self.print_solution = False
        self.print_truncation = False

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
        Test feasibility of optimization under assumption of no interaction.

        Assuming a hypothesis of no interaction (k_reg = 0) construct constraints
        using the birth-death model structure and confidence intervals on the 
        distribution of observed data. Solve to determine if the optimization is
        feasible: and so the data fits a model of no interaction, or infeasible:
        suggesting interaction is present or the data does not fit a birth-death
        model.

        Args:
            bounds: dictionary of confidence interval bounds computed by the 
                    bootstrap on the distribution of a sample from the dataset
            dataset: instance of Dataset class containing capture efficiency
                     and truncation settings used to construct constraints

        Returns:
            A dictionary containing results

            'status': GUROBI optimization status, typically 'OPTIMAL', 'INFEASIBLE'
                      or 'TIME_LIMIT' (see status codes below / in docs)
            'time': optimization solver runtime
        '''

        # if provided load WLS license credentials
        if self.licence_file:
            environment_parameters = json.load(open(self.license_file))
        # otherwise use default environment (e.g Named User license)
        else:
            environment_parameters = None

        # constraint settings: default to only joint constraints
        if self.constraint_settings is None:
            constraint_settings = {
                'bivariateB': True,
                'univariateB': False,
                'bivariateCME': True,
                'univariateCME': False
            }

        # silence output
        if self.silent:
            environment_parameters['OutputFlag'] = 0
        
        # environment context
        with gp.Env(params=environment_parameters) as env:

            # model context
            with gp.Model('birth-death-interaction-hypothesis', env=env) as md:

                # set optimization time limit
                md.Params.TimeLimit = self.time_limit

                # State space truncations

                # observed truncations: computed during bootstrap
                min_x1_OB = bounds['min_x1_OB']
                max_x1_OB = bounds['max_x1_OB']
                min_x2_OB = bounds['min_x2_OB']
                max_x2_OB = bounds['max_x2_OB']

                # marginal observed truncations
                minM_x1_OB = bounds['minM_x1_OB']
                maxM_x1_OB = bounds['maxM_x1_OB']
                minM_x2_OB = bounds['minM_x2_OB']
                maxM_x2_OB = bounds['maxM_x2_OB']

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

                # for each x1 marginal state used
                for x1_OB in range(minM_x1_OB, maxM_x1_OB + 1):

                    try:
                        # lookup original truncation
                        minM_x1_OG, maxM_x1_OG = dataset.truncationsM[f'{x1_OB}']

                    except KeyError:
                        # compute if not available
                        minM_x1_OG, maxM_x1_OG = truncation.compute_state_truncM(x1_OB, dataset.beta, dataset.threshM_OG)

                        # store
                        dataset.truncationsM[f'{x1_OB}'] = (minM_x1_OG, maxM_x1_OG)

                    # update overall min and max
                    if maxM_x1_OG > overall_max_x1_OG:
                        overall_max_x1_OG = maxM_x1_OG
                    if minM_x1_OG < overall_min_x1_OG:
                        overall_min_x1_OG = minM_x1_OG

                # for each x2 marginal state used
                for x2_OB in range(minM_x2_OB, maxM_x2_OB + 1):

                    try:
                        # lookup original truncation
                        minM_x2_OG, maxM_x2_OG = dataset.truncationsM[f'{x2_OB}']

                    except KeyError:
                        # compute if not available
                        minM_x2_OG, maxM_x2_OG = truncation.compute_state_truncM(x2_OB, dataset.beta, dataset.threshM_OG)

                        # store
                        dataset.truncationsM[f'{x2_OB}'] = (minM_x2_OG, maxM_x2_OG)

                    # update overall min and max
                    if maxM_x2_OG > overall_max_x2_OG:
                        overall_max_x2_OG = maxM_x2_OG
                    if minM_x2_OG < overall_min_x2_OG:
                        overall_min_x2_OG = minM_x2_OG
                
                if self.print_truncation:
                    print(f"Observed counts: [{min_x1_OB}, {max_x1_OB}] x [{min_x2_OB}, {max_x2_OB}]")
                    print(f"Original counts: [{overall_min_x1_OG}, {overall_max_x1_OG}] x [{overall_min_x2_OG}, {overall_max_x2_OG}]")

                # variables

                # marginal stationary distributions: original counts (size = largest original state + 1)
                p1 = md.addMVar(shape=(overall_max_x1_OG + 1), vtype=GRB.CONTINUOUS, name="p1", lb=0, ub=1)
                p2 = md.addMVar(shape=(overall_max_x2_OG + 1), vtype=GRB.CONTINUOUS, name="p2", lb=0, ub=1)

                # dummy joint variable to avoid triple products (not supported by GUROBI): should be removed by presolve
                p_dummy = md.addMVar(shape=(overall_max_x1_OG + 1, overall_max_x2_OG + 1), vtype=GRB.CONTINUOUS, name="p_dummy", lb=0, ub=1)

                # aggressive presolve to (hopefully) remove dummy variables
                md.Params.Presolve = 2

                # reaction rate constants
                rate_names = ['k_tx_1', 'k_tx_2', 'k_deg_1', 'k_deg_2']
                rates = md.addVars(rate_names, vtype=GRB.CONTINUOUS, lb=0, ub=self.K, name=rate_names)

                # constraints

                # fix k_deg_1 = 1, k_deg = 2 for identifiability
                md.addConstr(rates['k_deg_1'] == 1)
                md.addConstr(rates['k_deg_2'] == 1)

                # distributional constraints
                md.addConstr(p1.sum() <= 1, name="Distribution x1")
                md.addConstr(p2.sum() <= 1, name="Distribution x2")

                if self.settings['bivariateB']:

                    # stationary distribution bounds: for each observed count pair
                    for x1_OB in range(min_x1_OB, max_x1_OB + 1):
                        for x2_OB in range(min_x2_OB, max_x2_OB + 1):
                            
                            # original truncation: lookup from pre-computed dict
                            min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = dataset.truncations[f'({x1_OB}, {x2_OB})']
                            
                            # sum over truncation range (INCLUSIVE): drop terms with coefficients < thresh
                            sum_expr = gp.quicksum([
                                truncation.B_coeff(x1_OB, x2_OB, x1_OG, x2_OG, beta) * p1[x1_OG] * p2[x2_OG]
                                for x1_OG in range(min_x1_OG, max_x1_OG + 1)
                                for x2_OG in range(min_x2_OG, max_x2_OG + 1)
                                if truncation.B_coeff(x1_OB, x2_OB, x1_OG, x2_OG, dataset.beta) >= dataset.thresh_OG
                            ])
                            
                            md.addConstr(sum_expr >= bounds['joint'][0, x1_OB, x2_OB], name=f"B lb {x1_OB}, {x2_OB}")
                            md.addConstr(sum_expr <= bounds['joint'][1, x1_OB, x2_OB], name=f"B ub {x1_OB}, {x2_OB}")
                
                if self.settings['univariateB']:

                    # marginal stationary distribution bounds: for each observed count
                    for x1_OB in range(minM_x1_OB, maxM_x1_OB + 1):

                        # original truncation: lookup from pre-computed dict
                        minM_x1_OG, maxM_x1_OG = dataset.truncationsM[f'{x1_OB}']

                        # sum over truncation range (INCLUSIVE)
                        sum_expr = gp.quicksum([truncation.BM_coeff(x1_OB, x1_OG, dataset.beta) * p1[x1_OG] for x1_OG in range(minM_x1_OG, maxM_x1_OG + 1)])

                        md.addConstr(sum_expr >= bounds['x1'][0, x1_OB], name=f"B marginal lb {x1_OB}")
                        md.addConstr(sum_expr <= bounds['x1'][1, x1_OB], name=f"B marginal ub {x1_OB}")

                    for x2_OB in range(minM_x2_OB, maxM_x2_OB + 1):

                        # original truncation: lookup from pre-computed dict
                        minM_x2_OG, maxM_x2_OG = dataset.truncationsM[f'{x2_OB}']

                        # sum over truncation range (INCLUSIVE)
                        sum_expr = gp.quicksum([truncation.BM_coeff(x2_OB, x2_OG, dataset.beta) * p2[x2_OG] for x2_OG in range(minM_x2_OG, maxM_x2_OG + 1)])

                        md.addConstr(sum_expr >= bounds['x2'][0, x2_OB], name=f"B marginal lb {x2_OB}")
                        md.addConstr(sum_expr <= bounds['x2'][1, x2_OB], name=f"B marginal ub {x2_OB}")

                if self.settings['bivariateCME']:

                    # equate dummy joint variable to product of marginals: all original states
                    for x1_OG in range(overall_max_x1_OG + 1):
                        for x2_OG in range(overall_max_x2_OG + 1):

                            md.addConstr(p_dummy[x1_OG, x2_OG] == p1[x1_OG] * p2[x2_OG], name=f"Dummy joint definition {x1_OG}, {x2_OG}")

                    # CME: use dummy joint variable to avoid triple products: k_[] * p1[] * p2[]
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
                                rates['k_tx_1'] * x1_zero * p_dummy[x1_OG - 1, x2_OG] + \
                                rates['k_tx_2'] * x2_zero * p_dummy[x1_OG, x2_OG - 1] + \
                                rates['k_deg_1'] * (x1_OG + 1) * p_dummy[x1_OG + 1, x2_OG] + \
                                rates['k_deg_2'] * (x2_OG + 1) * p_dummy[x1_OG, x2_OG + 1] - \
                                (rates['k_tx_1'] + rates['k_tx_2'] + \
                                rates['k_deg_1'] * x1_OG + rates['k_deg_2'] * x2_OG) * p_dummy[x1_OG, x2_OG] == 0,
                                name=f"CME {x1_OG}, {x2_OG}"
                                )
                
                if self.settings['univariateCME']:

                    # CME for x1
                    for x1_OG in range(overall_max_x1_OG):
                        if x1_OG == 0:
                            x1_zero = 0
                        else:
                            x1_zero = 1

                        md.addConstr(
                            rates['k_tx_1'] * x1_zero * p1[x1_OG - 1] + \
                            rates['k_deg_1'] * (x1_OG + 1) * p1[x1_OG + 1] - \
                            (rates['k_tx_1'] + rates['k_deg_1'] * x1_OG) * p1[x1_OG] == 0,
                            name=f"Marginal CME x1 {x1_OG}"
                        )

                    # CME for x2
                    for x2_OG in range(overall_max_x2_OG):
                        if x2_OG == 0:
                            x2_zero = 0
                        else:
                            x2_zero = 1

                        md.addConstr(
                            rates['k_tx_2'] * x2_zero * p2[x2_OG - 1] + \
                            rates['k_deg_2'] * (x2_OG + 1) * p2[x2_OG + 1] - \
                            (rates['k_tx_2'] + rates['k_deg_2'] * x2_OG) * p2[x2_OG] == 0,
                            name=f"Marginal CME x2 {x2_OG}"
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

                # result
                solution = {}

                # testing feasibility: simply optimize 0
                md.setObjective(0, GRB.MINIMIZE)

                # set parameter (prevents 'infeasible or unbounded' ambiguity)
                md.Params.DualReductions = 0

                # set solution limit (stop after finding 1 feasible solution)
                md.Params.SolutionLimit = 1

                # test feasibility
                try:
                    md.optimize()
                    status_code = md.status
                except:
                    status_code = md.status

                # store status
                solution['status'] = status_codes[status_code]

                # store optimization times
                solution['time'] = md.Runtime

        # print
        if self.print_solution:
            print(f"Optimization status: {solution['status']} \nRuntime: {solution['time']}")

        return solution