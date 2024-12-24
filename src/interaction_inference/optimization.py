'''
Module for optimization inference methods

Given bootstrap confidence intervals (and truncations) from a sample, construct
constraints and optimize to detect if interaction is present or not.

Typical example:

# 'bounds' dictionary computed using bootstrap

# apply optimization method
results = optimization_hyp(bounds, 0.5)
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import gurobipy as gp
from gurobipy import GRB
import json
import tqdm

# ------------------------------------------------
# Functions
# ------------------------------------------------

def optimization_hyp(bounds, beta, settings=None, time_limit=300, silent=True,
                     K=100, print_solution=True, print_truncation=True,
                     truncations={}, truncationsM={}, thresh_OG=10**-6, threshM_OG=10**-6):
    '''
    Test feasibility of optimization under assumption of no interaction.

    Assuming a hypothesis of no interaction (k_reg = 0) construct constraints
    using the birth-death model structure and confidence intervals on the 
    distribution of observed data. Solve to determine if the optimization is
    feasible: and so the data fits a model of no interaction, or infeasible:
    suggesting interaction is present or the data does not fit a birth-death
    model.

    Args:
        bounds: dictionary of confidence interval and truncation information
                returned by the bootstrap
        beta: per cell capture efficiency vector / constant value
        settings: dictionary of constraint settings
        time_limit: optimization time limit before termination
        silent: toggle logging of optimization progress
        K: upper bound on reaction rates (default 100)
        print_solution: toggle printing final solution
        print_truncation: toggle printing truncation information
        truncations: dictionary of truncations (computed if not provided)
        truncationsM: dictionary of marginal truncations (computed if not provided)
        thresh_OG: truncation threshold
        threshM_OG: marginal truncation threshold

    Returns:
        A dictionary containing results:
        'status': GUROBI optimization status, typically 'OPTIMAL', 'INFEASIBLE'
                  or 'TIME_LIMIT' (see status codes below / in docs)
        'time': optimization solver runtime
    '''

    # constraint settings: default to joint constraints
    if settings is None:
        settings = {
            'bivariateB': True,
            'univariateB': False,
            'bivariateCME': True,
            'univariateCME': False
        }

    # create model
    md = gp.Model('birth-death-interaction-hyp')

    # set options
    if silent:
        md.Params.LogToConsole = 0

    # set time limit: 5 minute default
    md.Params.TimeLimit = time_limit

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
                min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = truncations[f'({x1_OB}, {x2_OB})']

            except KeyError:
                # compute if not available
                min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = findTrunc(x1_OB, x2_OB, beta, thresh_OG)

                # store
                truncations[f'({x1_OB}, {x2_OB})'] = (min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG)

                # store symmetry
                truncations[f'({x2_OB}, {x1_OB})'] = (min_x2_OG, max_x2_OG, min_x1_OG, max_x1_OG)

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
            minM_x1_OG, maxM_x1_OG = truncationsM[f'{x1_OB}']

        except KeyError:
            # compute if not available
            minM_x1_OG, maxM_x1_OG = findTruncM(x1_OB, beta, threshM_OG)

            # store
            truncationsM[f'{x1_OB}'] = (minM_x1_OG, maxM_x1_OG)

        # update overall min and max
        if maxM_x1_OG > overall_max_x1_OG:
            overall_max_x1_OG = maxM_x1_OG
        if minM_x1_OG < overall_min_x1_OG:
            overall_min_x1_OG = minM_x1_OG

    # for each x2 marginal state used
    for x2_OB in range(minM_x2_OB, maxM_x2_OB + 1):

        try:
            # lookup original truncation
            minM_x2_OG, maxM_x2_OG = truncationsM[f'{x2_OB}']

        except KeyError:
            # compute if not available
            minM_x2_OG, maxM_x2_OG = findTruncM(x2_OB, beta, threshM_OG)

            # store
            truncationsM[f'{x2_OB}'] = (minM_x2_OG, maxM_x2_OG)

        # update overall min and max
        if maxM_x2_OG > overall_max_x2_OG:
            overall_max_x2_OG = maxM_x2_OG
        if minM_x2_OG < overall_min_x2_OG:
            overall_min_x2_OG = minM_x2_OG
    
    if print_truncation:
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
    rates = md.addVars(rate_names, vtype=GRB.CONTINUOUS, lb=0, ub=K, name=rate_names)

    # constraints

    # fix k_deg_1 = 1, k_deg = 2 for identifiability
    md.addConstr(rates['k_deg_1'] == 1)
    md.addConstr(rates['k_deg_2'] == 1)

    # distributional constraints
    md.addConstr(p1.sum() <= 1, name="Distribution x1")
    md.addConstr(p2.sum() <= 1, name="Distribution x2")

    if settings['bivariateB']:

        # stationary distribution bounds: for each observed count pair
        for x1_OB in range(min_x1_OB, max_x1_OB + 1):
            for x2_OB in range(min_x2_OB, max_x2_OB + 1):
                
                # original truncation: lookup from pre-computed dict
                min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = truncations[f'({x1_OB}, {x2_OB})']
                
                # sum over truncation range (INCLUSIVE): drop terms with coefficients < thresh
                sum_expr = gp.quicksum([
                    B(x1_OB, x2_OB, x1_OG, x2_OG, beta) * p1[x1_OG] * p2[x2_OG]
                    for x1_OG in range(min_x1_OG, max_x1_OG + 1)
                    for x2_OG in range(min_x2_OG, max_x2_OG + 1)
                    if B(x1_OB, x2_OB, x1_OG, x2_OG, beta) >= thresh_OG
                ])
                
                md.addConstr(sum_expr >= bounds['joint'][0, x1_OB, x2_OB], name=f"B lb {x1_OB}, {x2_OB}")
                md.addConstr(sum_expr <= bounds['joint'][1, x1_OB, x2_OB], name=f"B ub {x1_OB}, {x2_OB}")
    
    if settings['univariateB']:

        # marginal stationary distribution bounds: for each observed count
        for x1_OB in range(minM_x1_OB, maxM_x1_OB + 1):

            # original truncation: lookup from pre-computed dict
            minM_x1_OG, maxM_x1_OG = truncationsM[f'{x1_OB}']

            # sum over truncation range (INCLUSIVE)
            sum_expr = gp.quicksum([BM(x1_OB, x1_OG, beta) * p1[x1_OG] for x1_OG in range(minM_x1_OG, maxM_x1_OG + 1)])

            md.addConstr(sum_expr >= bounds['x1'][0, x1_OB], name=f"B marginal lb {x1_OB}")
            md.addConstr(sum_expr <= bounds['x1'][1, x1_OB], name=f"B marginal ub {x1_OB}")

        for x2_OB in range(minM_x2_OB, maxM_x2_OB + 1):

            # original truncation: lookup from pre-computed dict
            minM_x2_OG, maxM_x2_OG = truncationsM[f'{x2_OB}']

            # sum over truncation range (INCLUSIVE)
            sum_expr = gp.quicksum([BM(x2_OB, x2_OG, beta) * p2[x2_OG] for x2_OG in range(minM_x2_OG, maxM_x2_OG + 1)])

            md.addConstr(sum_expr >= bounds['x2'][0, x2_OB], name=f"B marginal lb {x2_OB}")
            md.addConstr(sum_expr <= bounds['x2'][1, x2_OB], name=f"B marginal ub {x2_OB}")

    if settings['bivariateCME']:

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
    
    if settings['univariateCME']:

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
    if print_solution:
        print(f"Optimization status: {solution['status']} \nRuntime: {solution['time']}")

    return solution

# ------------------------------------------------

def optimization_hyp_WLS(licence_file, bounds, beta, settings=None, time_limit=300, silent=True,
                         K=100, print_solution=True, print_truncation=True,
                         truncations={}, truncationsM={}, thresh_OG=10**-6, threshM_OG=10**-6):
    '''
    Implementation of 'optimization_hyp' for users with a GUROBI WLS license.

    Assuming a hypothesis of no interaction (k_reg = 0) construct constraints
    using the birth-death model structure and confidence intervals on the 
    distribution of observed data. Solve to determine if the optimization is
    feasible: and so the data fits a model of no interaction, or infeasible:
    suggesting interaction is present or the data does not fit a birth-death
    model.

    Args:
        licence_file: location of .json file containing WLS license credentials
                     (DO NOT INCLUDE LICENSE INFORMATION IN PUBLIC REPOSITORY)
        bounds: dictionary of confidence interval and truncation information
                returned by the bootstrap
        beta: per cell capture efficiency vector / constant value
        settings: dictionary of constraint settings
        time_limit: optimization time limit before termination
        silent: toggle logging of optimization progress
        K: upper bound on reaction rates (default 100)
        print_solution: toggle printing final solution
        print_truncation: toggle printing truncation information
        truncations: dictionary of truncations (computed if not provided)
        truncationsM: dictionary of marginal truncations (computed if not provided)
        thresh_OG: truncation threshold
        threshM_OG: marginal truncation threshold

    Returns:
        A dictionary containing results:
        'status': GUROBI optimization status, typically 'OPTIMAL', 'INFEASIBLE'
                  or 'TIME_LIMIT' (see status codes below / in docs)
        'time': optimization solver runtime
    '''

    # load WLS license credentials
    options = json.load(open(license_file))

    # constraint settings: default to joint constraints
    if settings is None:
        settings = {
            'bivariateB': True,
            'univariateB': False,
            'bivariateCME': True,
            'univariateCME': False
        }

    # silence output
    if silent:
        options['OutputFlag'] = 0
    
    # environment context
    with gp.Env(params=options) as env:

        # model context
        with gp.Model('birth-death-interaction-hyp', env=env) as md:

            # set time limit: 5 minute default
            md.Params.TimeLimit = time_limit

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
                        min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = truncations[f'({x1_OB}, {x2_OB})']

                    except KeyError:
                        # compute if not available
                        min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = findTrunc(x1_OB, x2_OB, beta, thresh_OG)

                        # store
                        truncations[f'({x1_OB}, {x2_OB})'] = (min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG)

                        # store symmetry
                        truncations[f'({x2_OB}, {x1_OB})'] = (min_x2_OG, max_x2_OG, min_x1_OG, max_x1_OG)

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
                    minM_x1_OG, maxM_x1_OG = truncationsM[f'{x1_OB}']

                except KeyError:
                    # compute if not available
                    minM_x1_OG, maxM_x1_OG = findTruncM(x1_OB, beta, threshM_OG)

                    # store
                    truncationsM[f'{x1_OB}'] = (minM_x1_OG, maxM_x1_OG)

                # update overall min and max
                if maxM_x1_OG > overall_max_x1_OG:
                    overall_max_x1_OG = maxM_x1_OG
                if minM_x1_OG < overall_min_x1_OG:
                    overall_min_x1_OG = minM_x1_OG

            # for each x2 marginal state used
            for x2_OB in range(minM_x2_OB, maxM_x2_OB + 1):

                try:
                    # lookup original truncation
                    minM_x2_OG, maxM_x2_OG = truncationsM[f'{x2_OB}']

                except KeyError:
                    # compute if not available
                    minM_x2_OG, maxM_x2_OG = findTruncM(x2_OB, beta, threshM_OG)

                    # store
                    truncationsM[f'{x2_OB}'] = (minM_x2_OG, maxM_x2_OG)

                # update overall min and max
                if maxM_x2_OG > overall_max_x2_OG:
                    overall_max_x2_OG = maxM_x2_OG
                if minM_x2_OG < overall_min_x2_OG:
                    overall_min_x2_OG = minM_x2_OG
            
            if print_truncation:
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
            rates = md.addVars(rate_names, vtype=GRB.CONTINUOUS, lb=0, ub=K, name=rate_names)

            # constraints

            # fix k_deg_1 = 1, k_deg = 2 for identifiability
            md.addConstr(rates['k_deg_1'] == 1)
            md.addConstr(rates['k_deg_2'] == 1)

            # distributional constraints
            md.addConstr(p1.sum() <= 1, name="Distribution x1")
            md.addConstr(p2.sum() <= 1, name="Distribution x2")

            if settings['bivariateB']:

                # stationary distribution bounds: for each observed count pair
                for x1_OB in range(min_x1_OB, max_x1_OB + 1):
                    for x2_OB in range(min_x2_OB, max_x2_OB + 1):
                        
                        # original truncation: lookup from pre-computed dict
                        min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = truncations[f'({x1_OB}, {x2_OB})']
                        
                        # sum over truncation range (INCLUSIVE): drop terms with coefficients < thresh
                        sum_expr = gp.quicksum([
                            B(x1_OB, x2_OB, x1_OG, x2_OG, beta) * p1[x1_OG] * p2[x2_OG]
                            for x1_OG in range(min_x1_OG, max_x1_OG + 1)
                            for x2_OG in range(min_x2_OG, max_x2_OG + 1)
                            if B(x1_OB, x2_OB, x1_OG, x2_OG, beta) >= thresh_OG
                        ])
                        
                        md.addConstr(sum_expr >= bounds['joint'][0, x1_OB, x2_OB], name=f"B lb {x1_OB}, {x2_OB}")
                        md.addConstr(sum_expr <= bounds['joint'][1, x1_OB, x2_OB], name=f"B ub {x1_OB}, {x2_OB}")
            
            if settings['univariateB']:

                # marginal stationary distribution bounds: for each observed count
                for x1_OB in range(minM_x1_OB, maxM_x1_OB + 1):

                    # original truncation: lookup from pre-computed dict
                    minM_x1_OG, maxM_x1_OG = truncationsM[f'{x1_OB}']

                    # sum over truncation range (INCLUSIVE)
                    sum_expr = gp.quicksum([BM(x1_OB, x1_OG, beta) * p1[x1_OG] for x1_OG in range(minM_x1_OG, maxM_x1_OG + 1)])

                    md.addConstr(sum_expr >= bounds['x1'][0, x1_OB], name=f"B marginal lb {x1_OB}")
                    md.addConstr(sum_expr <= bounds['x1'][1, x1_OB], name=f"B marginal ub {x1_OB}")

                for x2_OB in range(minM_x2_OB, maxM_x2_OB + 1):

                    # original truncation: lookup from pre-computed dict
                    minM_x2_OG, maxM_x2_OG = truncationsM[f'{x2_OB}']

                    # sum over truncation range (INCLUSIVE)
                    sum_expr = gp.quicksum([BM(x2_OB, x2_OG, beta) * p2[x2_OG] for x2_OG in range(minM_x2_OG, maxM_x2_OG + 1)])

                    md.addConstr(sum_expr >= bounds['x2'][0, x2_OB], name=f"B marginal lb {x2_OB}")
                    md.addConstr(sum_expr <= bounds['x2'][1, x2_OB], name=f"B marginal ub {x2_OB}")

            if settings['bivariateCME']:

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
            
            if settings['univariateCME']:

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
    if print_solution:
        print(f"Optimization status: {solution['status']} \nRuntime: {solution['time']}")

    return solution


# ------------------------------------------------

def optimization_min(bounds, beta, settings=None, time_limit=300, silent=True,
                     K=100, print_solution=True, print_truncation=True,
                     truncations={}, truncationsM={}, thresh_OG=10**-6, threshM_OG=10**-6,
                     MIPGap=0.05, BestBdThresh=0.0001):
    '''
    Optimize lower bound on interaction parameter of birth-death model.

    Optimize lower bound on the interaction parameter k_reg using constraints
    from the birth-death model structure and confidence intervals on the 
    distribution of observed data. Solve to determine if the lower bound is
    zero: and so the data is consistent with a model of no interaction, or 
    non-zero: suggesting interaction is present.

    Args:
        bounds: dictionary of confidence interval and truncation information
                returned by the bootstrap
        beta: per cell capture efficiency vector / constant value
        settings: dictionary of constraint settings
        time_limit: optimization time limit before termination
        silent: toggle logging of optimization progress
        K: upper bound on reaction rates (default 100)
        print_solution: toggle printing final solution
        print_truncation: toggle printing truncation information
        truncations: dictionary of truncations (computed if not provided)
        truncationsM: dictionary of marginal truncations (computed if not provided)
        thresh_OG: truncation threshold
        threshM_OG: marginal truncation threshold
        MIPGap: GUROBI parameter for the optimal gap required to terminate
        BestBdThresh: threshold required to conclude non-zero lower bound on k_reg

    Returns:
        A dictionary containing results:
        'bound': lower bound on k_reg
        'status': GUROBI optimization status, typically 'OPTIMAL', 'USER_OBJ_LIMIT',
                  'INFEASIBLE' or 'TIME_LIMIT' (see status codes below / in docs)
        'time': optimization solver runtime
    '''

    # create model
    md = gp.Model('birth-death-interaction-hyp')

    # set options
    if silent:
        md.Params.LogToConsole = 0

    # set time limit: 5 minute default
    md.Params.TimeLimit = time_limit

    # optimization settings
    md.Params.MIPGap = MIPGap

    # aggressive presolve
    md.Params.Presolve = 2
    # focus on lower bound of objective: allows early termination
    md.Params.MIPFocus = 3

    # set threshold on BestBd for termination
    md.Params.BestBdStop = BestBdThresh

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
                min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = truncations[f'({x1_OB}, {x2_OB})']

            except KeyError:
                # compute if not available
                min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = findTrunc(x1_OB, x2_OB, beta, thresh_OG)

                # store
                truncations[f'({x1_OB}, {x2_OB})'] = (min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG)

                # store symmetry
                truncations[f'({x2_OB}, {x1_OB})'] = (min_x2_OG, max_x2_OG, min_x1_OG, max_x1_OG)

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

    if print_truncation:
        print(f"Observed counts: [{min_x1_OB}, {max_x1_OB}] x [{min_x2_OB}, {max_x2_OB}]")
        print(f"Original counts: [{overall_min_x1_OG}, {overall_max_x1_OG}] x [{overall_min_x2_OG}, {overall_max_x2_OG}]")

    # variables

    # stationary distribution: original counts (size = largest truncation)
    p = md.addMVar(shape=(overall_max_x1_OG + 1, overall_max_x2_OG + 1), vtype=GRB.CONTINUOUS, name="p", lb=0, ub=1)

    # reaction rate constants
    rate_names = ['k_tx_1', 'k_tx_2', 'k_deg_1', 'k_deg_2', 'k_reg']
    rates = md.addVars(rate_names, vtype=GRB.CONTINUOUS, lb=0, ub=K, name=rate_names)

    # constraints

    # fix k_deg_2 = 1 for identifiability
    md.addConstr(rates['k_deg_2'] == 1)

    # distribution
    md.addConstr(p.sum() <= 1, name="Distribution")

    # stationary distribution bounds: for each observed count pair
    for x1_OB in range(min_x1_OB, max_x1_OB + 1):
        for x2_OB in range(min_x2_OB, max_x2_OB + 1):
            
            # original truncation: lookup from pre-computed dict
            min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = truncations[f'({x1_OB}, {x2_OB})']
            
            # sum over truncation range (INCLUSIVE): drop terms with coefficients < thresh
            sum_expr = gp.quicksum([
                B(x1_OB, x2_OB, x1_OG, x2_OG, beta) * p1[x1_OG] * p2[x2_OG]
                for x1_OG in range(min_x1_OG, max_x1_OG + 1)
                for x2_OG in range(min_x2_OG, max_x2_OG + 1)
                if B(x1_OB, x2_OB, x1_OG, x2_OG, beta) >= thresh_OG
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
    if print_solution:
        print(f"k_reg lower bound: {solution['bound']}")
        print(f"Optimization status: {solution['status']}")
        print(f"Runtime: {solution['time']}")

    return solution

# ------------------------------------------------

def optimization_min_WLS(license_file, bounds, beta, settings=None, time_limit=300, silent=True,
                         K=100, print_solution=True, print_truncation=True,
                         truncations={}, truncationsM={}, thresh_OG=10**-6, threshM_OG=10**-6,
                         MIPGap=0.05, BestBdThresh=0.0001):
        
    '''
    Implementation of 'optimization_min' for users with a GUROBI WLS license.

    Optimize lower bound on the interaction parameter k_reg using constraints
    from the birth-death model structure and confidence intervals on the 
    distribution of observed data. Solve to determine if the lower bound is
    zero: and so the data is consistent with a model of no interaction, or 
    non-zero: suggesting interaction is present.

    Args:
        licence_file: location of .json file containing WLS license credentials
                    (DO NOT INCLUDE LICENSE INFORMATION IN PUBLIC REPOSITORY)
        bounds: dictionary of confidence interval and truncation information
                returned by the bootstrap
        beta: per cell capture efficiency vector / constant value
        settings: dictionary of constraint settings
        time_limit: optimization time limit before termination
        silent: toggle logging of optimization progress
        K: upper bound on reaction rates (default 100)
        print_solution: toggle printing final solution
        print_truncation: toggle printing truncation information
        truncations: dictionary of truncations (computed if not provided)
        truncationsM: dictionary of marginal truncations (computed if not provided)
        thresh_OG: truncation threshold
        threshM_OG: marginal truncation threshold
        MIPGap: GUROBI parameter for the optimal gap required to terminate
        BestBdThresh: threshold required to conclude non-zero lower bound on k_reg

    Returns:
        A dictionary containing results:
        'bound': lower bound on k_reg
        'status': GUROBI optimization status, typically 'OPTIMAL', 'USER_OBJ_LIMIT',
                  'INFEASIBLE' or 'TIME_LIMIT' (see status codes below / in docs)
        'time': optimization solver runtime
    '''

    # load WLS license credentials
    options = json.load(open("D:/Projects/ProjectPaper/WLS_credentials.json"))

    # silence output
    if silent:
        options['OutputFlag'] = 0
    
    # environment context
    with gp.Env(params=options) as env:

        # model context
        with gp.Model('birth-death-regulation-capture-efficiency-min', env=env) as md:

            # set time limit: 5 minute default
            md.Params.TimeLimit = time_limit

            # optimization settings
            md.Params.MIPGap = MIPGap

            # aggressive presolve
            md.Params.Presolve = 2
            # focus on lower bound of objective: allows early termination
            md.Params.MIPFocus = 3

            # set threshold on BestBd for termination
            md.Params.BestBdStop = BestBdThresh

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
                        min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = truncations[f'({x1_OB}, {x2_OB})']

                    except KeyError:
                        # compute if not available
                        min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = findTrunc(x1_OB, x2_OB, beta, thresh_OG)

                        # store
                        truncations[f'({x1_OB}, {x2_OB})'] = (min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG)

                        # store symmetry
                        truncations[f'({x2_OB}, {x1_OB})'] = (min_x2_OG, max_x2_OG, min_x1_OG, max_x1_OG)

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

            if print_truncation:
                print(f"Observed counts: [{min_x1_OB}, {max_x1_OB}] x [{min_x2_OB}, {max_x2_OB}]")
                print(f"Original counts: [{overall_min_x1_OG}, {overall_max_x1_OG}] x [{overall_min_x2_OG}, {overall_max_x2_OG}]")

            # variables

            # stationary distribution: original counts (size = largest truncation)
            p = md.addMVar(shape=(overall_max_x1_OG + 1, overall_max_x2_OG + 1), vtype=GRB.CONTINUOUS, name="p", lb=0, ub=1)

            # reaction rate constants
            rate_names = ['k_tx_1', 'k_tx_2', 'k_deg_1', 'k_deg_2', 'k_reg']
            rates = md.addVars(rate_names, vtype=GRB.CONTINUOUS, lb=0, ub=K, name=rate_names)

            # constraints

            # fix k_deg_2 = 1 for identifiability
            md.addConstr(rates['k_deg_2'] == 1)

            # distribution
            md.addConstr(p.sum() <= 1, name="Distribution")

            # stationary distribution bounds: for each observed count pair
            for x1_OB in range(min_x1_OB, max_x1_OB + 1):
                for x2_OB in range(min_x2_OB, max_x2_OB + 1):
                    
                    # original truncation: lookup from pre-computed dict
                    min_x1_OG, max_x1_OG, min_x2_OG, max_x2_OG = truncations[f'({x1_OB}, {x2_OB})']
                    
                    # sum over truncation range (INCLUSIVE): drop terms with coefficients < thresh
                    sum_expr = gp.quicksum([
                        B(x1_OB, x2_OB, x1_OG, x2_OG, beta) * p1[x1_OG] * p2[x2_OG]
                        for x1_OG in range(min_x1_OG, max_x1_OG + 1)
                        for x2_OG in range(min_x2_OG, max_x2_OG + 1)
                        if B(x1_OB, x2_OB, x1_OG, x2_OG, beta) >= thresh_OG
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
    if print_solution:
        print(f"k_reg lower bound: {solution['bound']}")
        print(f"Optimization status: {solution['status']}")
        print(f"Runtime: {solution['time']}")

    return solution
