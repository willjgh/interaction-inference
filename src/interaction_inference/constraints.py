'''
Module for functions to add constraints to optimization model
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# ------------------------------------------------
# Variables
# ------------------------------------------------

def add_variables(optimization, model, i):

    # stage variables to be added
    staged_variables = set()

    # base constraints
    if "k_deg_1" in optimization.constraints:
        staged_variables.update(['k_deg_1'])
    if "k_deg_2" in optimization.constraints:
        staged_variables.update(['k_deg_2'])
    if "k_reg" in optimization.constraints:
        staged_variables.update(['k_reg'])

    # B method birth-death constraints
    if "factorization" in optimization.constraints:
        staged_variables.update(['p', 'p1', 'p2'])
    if "joint_probability" in optimization.constraints:
        staged_variables.update(['p'])
    if "probability" in optimization.constraints:
        staged_variables.update(['p1', 'p2'])
    if "marginal_probability_1" in optimization.constraints:
        staged_variables.update(['p1'])
    if "marginal_probability_2" in optimization.constraints:
        staged_variables.update(['p2'])
    if "CME" in optimization.constraints:
        staged_variables.update(['p', 'k_tx_1', 'k_tx_2', 'k_deg_1', 'k_deg_2', 'k_reg'])
    if "marginal_CME_1" in optimization.constraints:
        staged_variables.update(['p1', 'k_tx_1', 'k_deg_1'])
    if "marginal_CME_2" in optimization.constraints:
        staged_variables.update(['p2', 'k_tx_2', 'k_deg_2'])

    # B method telegraph constraints
    if "marginal_CME_TE" in optimization.constraints:
        staged_variables.update(['pg1', 'pg2', 'k_on_1', 'k_on_2', 'k_off_1', 'k_off_2', 'k_tx_1', 'k_tx_2', 'k_deg_1', 'k_deg_2'])
    if "TE_equality" in optimization.constraints:
        staged_variables.update(['p1', 'p2', 'pg1', 'pg2'])

    # Moment constraints
    if "moment" in optimization.constraints:
        staged_variables.update(['p1', 'p2', 'E_x1', 'E_x2'])
    if "higher_moment" in optimization.constraints:
        staged_variables.update(['p1', 'p2'])
    if "dummy_moment" in optimization.constraints:
        staged_variables.update(['E_x1', 'E_x2'])

    # downsampled constraints
    if "downsampled_probability" in optimization.constraints:
        staged_variables.update(['pd'])
    if "downsampled_marginal_probability_1" in optimization.constraints:
        staged_variables.update(['pd1'])
    if "downsampled_marginal_probability_2" in optimization.constraints:
        staged_variables.update(['pd2'])
    if "downsampled_CME" in optimization.constraints:
        staged_variables.update(['pd', 'fm', 'k_tx_1', 'k_tx_2', 'k_deg_1', 'k_deg_2', 'k_reg'])
    if "downsampled_marginal_CME_1" in optimization.constraints:
        staged_variables.update(['pd1', 'fm1', 'k_tx_1', 'k_deg_1'])
    if "downsampled_marginal_CME_2" in optimization.constraints:
        staged_variables.update(['pd2', 'fm2', 'k_tx_2', 'k_deg_2'])

    # variable dict
    variables = {}

    if 'p1' in staged_variables:
        variables['p1'] = model.addMVar(shape=(optimization.overall_extent_OG[f'sample-{i}']['max_x1_OG'] + 1), vtype=GRB.CONTINUOUS, name="p1", lb=0, ub=1)
        model.addConstr(variables['p1'].sum() <= 1, name="Dist_p1")
    if 'p2' in staged_variables:
        variables['p2'] = model.addMVar(shape=(optimization.overall_extent_OG[f'sample-{i}']['max_x2_OG'] + 1), vtype=GRB.CONTINUOUS, name="p2", lb=0, ub=1)
        model.addConstr(variables['p2'].sum() <= 1, name="Dist_p2")
    if 'pg1' in staged_variables:
        variables['pg1'] = model.addMVar(shape=(2*(optimization.overall_extent_OG[f'sample-{i}']['max_x1_OG'] + 1)), vtype=GRB.CONTINUOUS, name="pg1", lb=0, ub=1)
        model.addConstr(variables['pg1'].sum() <= 1, name="Dist_pg1")
    if 'pg2' in staged_variables:
        variables['pg2'] = model.addMVar(shape=(2*(optimization.overall_extent_OG[f'sample-{i}']['max_x2_OG'] + 1)), vtype=GRB.CONTINUOUS, name="pg2", lb=0, ub=1)
        model.addConstr(variables['pg2'].sum() <= 1, name="Dist_pg2")
    if 'p' in staged_variables:
        variables['p'] = model.addMVar(shape=(optimization.overall_extent_OG[f'sample-{i}']['max_x1_OG'] + 1, optimization.overall_extent_OG[f'sample-{i}']['max_x2_OG'] + 1), vtype=GRB.CONTINUOUS, name="p", lb=0, ub=1)
        model.addConstr(variables['p'].sum() <= 1, name="Dist_p")
    
    if 'k_on_1' in staged_variables:
        variables['k_on_1'] = model.addVar(vtype=GRB.CONTINUOUS, name="k_on_1", lb=0, ub=optimization.K)
    if 'k_on_2' in staged_variables:
        variables['k_on_2'] = model.addVar(vtype=GRB.CONTINUOUS, name="k_on_2", lb=0, ub=optimization.K)
    if 'k_off_1' in staged_variables:
        variables['k_off_1'] = model.addVar(vtype=GRB.CONTINUOUS, name="k_off_1", lb=0, ub=optimization.K)
    if 'k_off_2' in staged_variables:
        variables['k_off_2'] = model.addVar(vtype=GRB.CONTINUOUS, name="k_off_2", lb=0, ub=optimization.K)
    if 'k_tx_1' in staged_variables:
        variables['k_tx_1'] = model.addVar(vtype=GRB.CONTINUOUS, name="k_tx_1", lb=0, ub=optimization.K)
    if 'k_tx_2' in staged_variables:
        variables['k_tx_2'] = model.addVar(vtype=GRB.CONTINUOUS, name="k_tx_2", lb=0, ub=optimization.K)
    if 'k_deg_1' in staged_variables:
        variables['k_deg_1'] = model.addVar(vtype=GRB.CONTINUOUS, name="k_deg_1", lb=0, ub=optimization.K)
    if 'k_deg_2' in staged_variables:
        variables['k_deg_2'] = model.addVar(vtype=GRB.CONTINUOUS, name="k_deg_2", lb=0, ub=optimization.K)
    if 'k_reg' in staged_variables:
        variables['k_reg'] = model.addVar(vtype=GRB.CONTINUOUS, name="k_reg", lb=0, ub=optimization.K)
    
    if 'E_x1' in staged_variables:
        variables['E_x1'] = model.addVar(vtype=GRB.CONTINUOUS, name="E_x1")
    if 'E_x2' in staged_variables:
        variables['E_x2'] = model.addVar(vtype=GRB.CONTINUOUS, name="E_x2")

    if 'pd1' in staged_variables:
        variables['pd1'] = model.addMVar(shape=(optimization.dataset.truncationM_OB[f'sample-{i}']['maxM_x1_OB'] + 1), vtype=GRB.CONTINUOUS, name="pd1", lb=0, ub=1)
        model.addConstr(variables['pd1'].sum() <= 1, name="Dist_pd1")
    if 'pd2' in staged_variables:
        variables['pd2'] = model.addMVar(shape=(optimization.dataset.truncationM_OB[f'sample-{i}']['maxM_x2_OB'] + 1), vtype=GRB.CONTINUOUS, name="pd2", lb=0, ub=1)
        model.addConstr(variables['pd2'].sum() <= 1, name="Dist_pd2")
    if 'pd' in staged_variables:
        variables['pd'] = model.addMVar(shape=(optimization.dataset.truncation_OB[f'sample-{i}']['max_x1_OB'] + 1, optimization.dataset.truncation_OB[f'sample-{i}']['max_x2_OB'] + 1), vtype=GRB.CONTINUOUS, name="pd", lb=0, ub=1)
        model.addConstr(variables['pd'].sum() <= 1, name="Dist_pd")
    
    if 'fm1' in staged_variables:
        variables['fm1'] = model.addMVar(shape=(optimization.dataset.truncationM_OB[f'sample-{i}']['maxM_x1_OB'] + 1), vtype=GRB.CONTINUOUS, name="fm1", lb=0, ub=1)
    if 'fm2' in staged_variables:
        variables['fm2'] = model.addMVar(shape=(optimization.dataset.truncationM_OB[f'sample-{i}']['maxM_x2_OB'] + 1), vtype=GRB.CONTINUOUS, name="fm2", lb=0, ub=1)
    if 'fm' in staged_variables:
        variables['fm'] = model.addMVar(shape=(optimization.dataset.truncation_OB[f'sample-{i}']['max_x1_OB'] + 1, optimization.dataset.truncation_OB[f'sample-{i}']['max_x2_OB'] + 1), vtype=GRB.CONTINUOUS, name="fm", lb=0, ub=1)
    
    return variables

# ------------------------------------------------
# Constraints
# ------------------------------------------------

def add_constraints(optimization, model, variables, i):

    # Base constraints
    if "k_deg_1" in optimization.constraints:
        add_k_deg_1_constraints(
            model,
            variables
        )
    if "k_deg_2" in optimization.constraints:
        add_k_deg_2_constraints(
            model,
            variables
        )
    if "k_reg" in optimization.constraints:
        add_k_reg_constraints(
            model,
            variables
        )
    
    # B method birth death constraints
    if "factorization" in optimization.constraints:
        add_factorization_constraints(
            model,
            variables
        )
    if "joint_probability" in optimization.constraints:
        add_joint_probability_constraints(
            model,
            variables,
            optimization.dataset.probs_OB[f'sample-{i}'],
            optimization.dataset.truncation_OB[f'sample-{i}'],
            optimization.dataset.truncation_OG,
            optimization.dataset.name
        )
    if "probability" in optimization.constraints:
        add_probability_constraints(
            model,
            variables,
            optimization.dataset.probs_OB[f'sample-{i}'],
            optimization.dataset.truncation_OB[f'sample-{i}'],
            optimization.dataset.truncation_OG,
            optimization.dataset.name
        )
    if "marginal_probability_1" in optimization.constraints:
        add_marginal_probability_1_constraints(
            model,
            variables,
            optimization.dataset.probs_OB[f'sample-{i}'],
            optimization.dataset.truncationM_OB[f'sample-{i}'],
            optimization.dataset.truncation_OG,
            optimization.dataset.name
        )
    if "marginal_probability_2" in optimization.constraints:
        add_marginal_probability_2_constraints(
            model,
            variables,
            optimization.dataset.probs_OB[f'sample-{i}'],
            optimization.dataset.truncationM_OB[f'sample-{i}'],
            optimization.dataset.truncation_OG,
            optimization.dataset.name
        )
    if "CME" in optimization.constraints:
        add_CME_constraints(
            model,
            variables,
            optimization.overall_extent_OG[f'sample-{i}']
        )
    if "marginal_CME_1" in optimization.constraints:
        add_marginal_CME_1_constraints(
            model,
            variables,
            optimization.overall_extent_OG[f'sample-{i}']
        )
    if "marginal_CME_2" in optimization.constraints:
        add_marginal_CME_2_constraints(
            model,
            variables,
            optimization.overall_extent_OG[f'sample-{i}']
        )

    # B method telegraph constraints
    if "marginal_CME_TE" in optimization.constraints:
        add_marginal_CME_TE_constraints(
            model,
            variables,
            optimization.overall_extent_OG[f'sample-{i}']
        )
    if "TE_equality" in optimization.constraints:
        add_TE_equality_constraints(
            model,
            variables,
            optimization.overall_extent_OG[f'sample-{i}']
        )

    # Moment constraints
    if "moment" in optimization.constraints:
        add_moment_constraints(
            model,
            variables,
            optimization.dataset.moment_extent_OG[f'sample-{i}'],
            optimization.dataset.moments_OB[f'sample-{i}'],
            optimization.dataset.beta
        )
    if "higher_moment" in optimization.constraints:
        add_higher_moment_constraints(
            model,
            variables,
            optimization.dataset.moment_extent_OG[f'sample-{i}'],
            optimization.dataset.moments_OB[f'sample-{i}'],
            optimization.dataset.beta
        )
    if "dummy_moment" in optimization.constraints:
        add_dummy_moment_constraints(
            model,
            variables,
            optimization.dataset.moments_OB[f'sample-{i}'],
            optimization.dataset.beta
        )

    # Downsampled constraints
    if "downsampled_probability" in optimization.constraints:
        add_downsampled_probability_constraints(
            model,
            variables,
            optimization.dataset.probs_OB[f'sample-{i}'],
            optimization.dataset.truncation_OB[f'sample-{i}']
        )
    if "downsampled_marginal_probability_1" in optimization.constraints:
        add_downsampled_marginal_probability_1_constraints(
            model,
            variables,
            optimization.dataset.probs_OB[f'sample-{i}'],
            optimization.dataset.truncationM_OB[f'sample-{i}']
        )
    if "downsampled_marginal_probability_2" in optimization.constraints:
        add_downsampled_marginal_probability_2_constraints(
            model,
            variables,
            optimization.dataset.probs_OB[f'sample-{i}'],
            optimization.dataset.truncationM_OB[f'sample-{i}']
        )
    if "downsampled_CME" in optimization.constraints:
        add_downsampled_CME_constraints(
            model,
            variables,
            optimization.dataset.fm_OB[f'sample-{i}'],
            optimization.dataset.truncation_OB[f'sample-{i}']
        )
    if "downsampled_marginal_CME_1" in optimization.constraints:
        add_downsampled_marginal_CME_1_constraints(
            model,
            variables,
            optimization.dataset.fm_OB[f'sample-{i}'],
            optimization.dataset.truncationM_OB[f'sample-{i}']
        )
    if "downsampled_marginal_CME_2" in optimization.constraints:
        add_downsampled_marginal_CME_2_constraints(
            model,
            variables,
            optimization.dataset.fm_OB[f'sample-{i}'],
            optimization.dataset.truncationM_OB[f'sample-{i}']
        )

# ------------------------------------------------
# Basic constraints
# ------------------------------------------------

def add_k_deg_1_constraints(model, variables):
    model.addConstr(variables['k_deg_1'] == 1, name="Fix_k_deg_1")

def add_k_deg_2_constraints(model, variables):
    model.addConstr(variables['k_deg_2'] == 1, name="Fix_k_deg_2")

def add_k_reg_constraints(model, variables):
    model.addConstr(variables['k_reg'] == 0, name="Fix_k_reg")

# ------------------------------------------------
# B method Birth Death constraints
# ------------------------------------------------

def add_factorization_constraints(model, variables):

    # get variables
    p1 = variables['p1']
    p2 = variables['p2']
    p = variables['p']

    # outer product marginals
    outer = p1[:, None] @ p2[None, :]

    # equate dummy joint variable to product of marginals: all original states
    model.addConstr(p == outer, name=f"Joint_factorize")

def add_joint_probability_constraints(model, variables, probs_OB, truncation_OB, truncation_OG, dataset_name):
    '''
    Joint probability constraints without independence
    Will be computationally impossible for all but high capture
    Only for comparison tests to downsampled data
    '''

    # get OB truncation for sample i
    min_x1_OB = truncation_OB['min_x1_OB']
    max_x1_OB = truncation_OB['max_x1_OB']
    min_x2_OB = truncation_OB['min_x2_OB']
    max_x2_OB = truncation_OB['max_x2_OB']
            
    # for each OB state pair in truncation
    for x1_OB in range(min_x1_OB, max_x1_OB + 1):
        for x2_OB in range(min_x2_OB, max_x2_OB + 1):

            # get OG truncation for OB state pair
            min_x1_OG, max_x1_OG = truncation_OG[x1_OB]
            min_x2_OG, max_x2_OG = truncation_OG[x2_OB]
            
            # load coefficient grid for OB state pair
            B_coeffs = np.load(f"./Temp/Coefficients/{dataset_name}-state-{x1_OB}-{x2_OB}.npy")

            # slice variables to truncation
            p_slice = variables['p'][min_x1_OG: max_x1_OG + 1, min_x2_OG: max_x2_OG + 1]

            # B matrix sum
            sum_expr = gp.quicksum(B_coeffs * p_slice)
        
            # form constraints using CI bounds
            model.addConstr(sum_expr >= probs_OB['bounds'][0, x1_OB, x2_OB], name=f"B_lb_{x1_OB}_{x2_OB}")
            model.addConstr(sum_expr <= probs_OB['bounds'][1, x1_OB, x2_OB], name=f"B_ub_{x1_OB}_{x2_OB}")

def add_probability_constraints(model, variables, probs_OB, truncation_OB, truncation_OG, dataset_name):

    # get OB truncation for sample i
    min_x1_OB = truncation_OB['min_x1_OB']
    max_x1_OB = truncation_OB['max_x1_OB']
    min_x2_OB = truncation_OB['min_x2_OB']
    max_x2_OB = truncation_OB['max_x2_OB']
            
    # for each OB state pair in truncation
    for x1_OB in range(min_x1_OB, max_x1_OB + 1):
        for x2_OB in range(min_x2_OB, max_x2_OB + 1):

            # get OG truncation for OB state pair
            min_x1_OG, max_x1_OG = truncation_OG[x1_OB]
            min_x2_OG, max_x2_OG = truncation_OG[x2_OB]
            
            # load coefficient grid for OB state pair
            B_coeffs = np.load(f"./Temp/Coefficients/{dataset_name}-state-{x1_OB}-{x2_OB}.npy")

            # slice variables to truncation
            p1_slice = variables['p1'][min_x1_OG: max_x1_OG + 1]
            p2_slice = variables['p2'][min_x2_OG: max_x2_OG + 1]

            # bilinear form
            sum_expr = p1_slice.T @ B_coeffs @ p2_slice
        
            # form constraints using CI bounds
            model.addConstr(sum_expr >= probs_OB['bounds'][0, x1_OB, x2_OB], name=f"B_lb_{x1_OB}_{x2_OB}")
            model.addConstr(sum_expr <= probs_OB['bounds'][1, x1_OB, x2_OB], name=f"B_ub_{x1_OB}_{x2_OB}")

def add_marginal_probability_1_constraints(model, variables, probs_OB, truncationM_OB, truncation_OG, dataset_name):

    # get marginal OB truncation for sample i
    minM_x1_OB = truncationM_OB['minM_x1_OB']
    maxM_x1_OB = truncationM_OB['maxM_x1_OB']

    # for each OB state in truncation
    for x1_OB in range(minM_x1_OB, maxM_x1_OB + 1):

        # get OG truncation
        min_x1_OG, max_x1_OG = truncation_OG[x1_OB]

        # load marginal coefficient array for OB state
        Bm_coeffs = np.load(f"./Temp/Coefficients/{dataset_name}-state-{x1_OB}.npy")

        # slice variable to truncation
        p1_slice = variables['p1'][min_x1_OG: max_x1_OG + 1]

        # linear expression of sum
        sum_expr = gp.quicksum(Bm_coeffs * p1_slice)

        # form constraints using CI bounds
        model.addConstr(sum_expr >= probs_OB['x1_bounds'][0, x1_OB], name=f"Bm_x1_lb_{x1_OB}")
        model.addConstr(sum_expr <= probs_OB['x1_bounds'][1, x1_OB], name=f"Bm_x1_ub_{x1_OB}")

def add_marginal_probability_2_constraints(model, variables, probs_OB, truncationM_OB, truncation_OG, dataset_name):

    # get marginal OB truncation for sample i
    minM_x2_OB = truncationM_OB['minM_x2_OB']
    maxM_x2_OB = truncationM_OB['maxM_x2_OB']

    # for each OB state in truncation
    for x2_OB in range(minM_x2_OB, maxM_x2_OB + 1):

        # get OG truncation
        min_x2_OG, max_x2_OG = truncation_OG[x2_OB]

        # load marginal coefficient array for OB state
        Bm_coeffs = np.load(f"./Temp/Coefficients/{dataset_name}-state-{x2_OB}.npy")

        # slice variable to truncation
        p2_slice = variables['p2'][min_x2_OG: max_x2_OG + 1]

        # linear expression of sum
        sum_expr = gp.quicksum(Bm_coeffs * p2_slice)

        # form constraints using CI bounds
        model.addConstr(sum_expr >= probs_OB['x2_bounds'][0, x2_OB], name=f"Bm_x2_lb_{x2_OB}")
        model.addConstr(sum_expr <= probs_OB['x2_bounds'][1, x2_OB], name=f"Bm_x2_ub_{x2_OB}")

def add_CME_constraints(model, variables, overall_extent_OG):

    # get extent of OG states
    max_x1_OG = overall_extent_OG['max_x1_OG']
    max_x2_OG = overall_extent_OG['max_x2_OG']

    # get variables
    p = variables['p']
    k_tx_1 = variables['k_tx_1']
    k_tx_2 = variables['k_tx_2']
    k_deg_1 = variables['k_deg_1']
    k_deg_2 = variables['k_deg_2']
    k_reg = variables['k_reg']
    
    # manually add x1_OG = x2_OG = 0 constraint (to avoid p(0) terms)
    model.addConstr(
        0 == k_deg_1 * p[1, 0] + \
        k_deg_2 * p[0, 1] + \
        k_reg * p[1, 1] - \
        (k_tx_1 + k_tx_2) * p[0, 0],
        name="CME_0_0"
    )

    # manually add x1_OG = 0 constraints (to avoid p1(-1) terms)
    model.addConstrs(
        (
            0 == k_tx_2 * p[0, x2_OG - 1] + \
            k_deg_1 * p[1, x2_OG] + \
            k_deg_2 * (x2_OG + 1) * p[0, x2_OG + 1] + \
            k_reg * (x2_OG + 1) * p[1, x2_OG + 1] - \
            (k_tx_1 + k_tx_2 + k_deg_2 * x2_OG) * p[0, x2_OG]
            for x2_OG in range(1, max_x2_OG)
        ),
        name="CME_0_x2"
    )
    # manually add x2_OG = 0 constraints (to avoid p2(-1) terms)
    model.addConstrs(
        (
            0 == k_tx_1 * p[x1_OG - 1, 0] + \
            k_deg_1 * (x1_OG + 1) * p[x1_OG + 1, 0] + \
            k_deg_2 * p[x1_OG, 1] + \
            k_reg * (x1_OG + 1) * p[x1_OG + 1, 1] - \
            (k_tx_1 + k_tx_2 + k_deg_1 * x1_OG) * p[x1_OG, 0]
            for x1_OG in range(1, max_x1_OG)
        ),
        name="CME_x1_0"
    )

    # add CME constraints
    model.addConstrs(
        (
            0 == k_tx_1 * p[x1_OG - 1, x2_OG] + \
            k_tx_2 * p[x1_OG, x2_OG - 1] + \
            k_deg_1 * (x1_OG + 1) * p[x1_OG + 1, x2_OG] + \
            k_deg_2 * (x2_OG + 1) * p[x1_OG, x2_OG + 1] + \
            k_reg * (x1_OG + 1) * (x2_OG + 1) * p[x1_OG + 1, x2_OG + 1] - \
            (k_tx_1 + k_tx_2 + k_deg_1 * x1_OG + k_deg_2 * x2_OG + k_reg * x1_OG * x2_OG) * p[x1_OG, x2_OG]
            for x1_OG in range(1, max_x1_OG)
            for x2_OG in range(1, max_x2_OG)
        ),
        name="CME_x1_x2"
    )

def add_marginal_CME_1_constraints(model, variables, overall_extent_OG):

    # get extent of OG states
    max_x1_OG = overall_extent_OG['max_x1_OG']

    # get variables
    p1 = variables['p1']
    k_tx_1 = variables['k_tx_1']
    k_deg_1 = variables['k_deg_1']

    # construct Q matrices: 1 more column than square to add upper diagonal to last row
    Q_tx_1 = (np.diag([1 for x in range(1, max_x1_OG + 1)], -1) - np.diag([1 for x in range(max_x1_OG + 1)]))[:-1, :]
    Q_deg_1 = (np.diag([x for x in range(1, max_x1_OG + 1)], 1) - np.diag([x for x in range(max_x1_OG + 1)]))[:-1, :]

    # add matrix constraints
    model.addConstr(
        k_tx_1 * (Q_tx_1 @ p1) + k_deg_1 * (Q_deg_1 @ p1) == 0,
        name="Marginal_CME_x1"
    )

def add_marginal_CME_2_constraints(model, variables, overall_extent_OG):

    # get extent of OG states
    max_x2_OG = overall_extent_OG['max_x2_OG']

    # get variables
    p2 = variables['p2']
    k_tx_2 = variables['k_tx_2']
    k_deg_2 = variables['k_deg_2']

    # construct Q matrices: 1 more column than square to add upper diagonal to last row
    Q_tx_2 = (np.diag([1 for x in range(1, max_x2_OG + 1)], -1) - np.diag([1 for x in range(max_x2_OG + 1)]))[:-1, :]
    Q_deg_2 = (np.diag([x for x in range(1, max_x2_OG + 1)], 1) - np.diag([x for x in range(max_x2_OG + 1)]))[:-1, :]

    # add matrix constraints
    model.addConstr(
        k_tx_2 * (Q_tx_2 @ p2) + k_deg_2 * (Q_deg_2 @ p2) == 0,
        name="Marginal_CME_x2"
    )

# ------------------------------------------------
# B method Telegraph constraints
# ------------------------------------------------

def add_marginal_CME_TE_constraints(model, variables, overall_extent_OG):

    # get extent of OG states
    max_x1_OG = overall_extent_OG['max_x1_OG']
    max_x2_OG = overall_extent_OG['max_x2_OG']

    # get variables
    pg1 = variables['pg1']
    pg2 = variables['pg2']
    k_on_1 = variables['k_on_1']
    k_on_2 = variables['k_on_2']
    k_off_1 = variables['k_off_1']
    k_off_2 = variables['k_off_2']
    k_tx_1 = variables['k_tx_1']
    k_tx_2 = variables['k_tx_2']
    k_deg_1 = variables['k_deg_1']
    k_deg_2 = variables['k_deg_2']

    # variable sizes
    N1 = 2*(max_x1_OG + 1)
    N2 = 2*(max_x2_OG + 1)

    # construct Q matrices
    Q_on_1 = (np.diag([0 if x % 2 else -1 for x in range(N1)]) + np.diag([0 if x % 2 else 1 for x in range(N1 - 1)], -1))[:-2, :]
    Q_on_2 = (np.diag([0 if x % 2 else -1 for x in range(N2)]) + np.diag([0 if x % 2 else 1 for x in range(N2 - 1)], -1))[:-2, :]

    Q_off_1 = (np.diag([-1 if x % 2 else 0 for x in range(N1)]) + np.diag([0 if x % 2 else 1 for x in range(N1 - 1)], 1))[:-2, :]
    Q_off_2 = (np.diag([-1 if x % 2 else 0 for x in range(N2)]) + np.diag([0 if x % 2 else 1 for x in range(N2 - 1)], 1))[:-2, :]

    Q_tx_1 = (np.diag([-1 if x % 2 else 0 for x in range(N1)]) + np.diag([1 if x % 2 else 0 for x in range(N1 - 2)], -2))[:-2, :]
    Q_tx_2 = (np.diag([-1 if x % 2 else 0 for x in range(N2)]) + np.diag([1 if x % 2 else 0 for x in range(N2 - 2)], -2))[:-2, :]

    deg_diag_1 = np.array([(x // 2) for x in range(N1)])
    deg_diag_2 = np.array([(x // 2) for x in range(N2)])

    Q_deg_1 = (np.diag(-deg_diag_1) + np.diag(deg_diag_1[2:], 2))[:-2, ]
    Q_deg_2 = (np.diag(-deg_diag_2) + np.diag(deg_diag_2[2:], 2))[:-2, ]

    # add matrix constraints
    model.addConstr(
        k_on_1 * (Q_on_1 @ pg1) + k_off_1 * (Q_off_1 @ pg1) + k_tx_1 * (Q_tx_1 @ pg1) + k_deg_1 * (Q_deg_1 @ pg1) == 0,
        name="Marginal_CME_x1"
    )

    model.addConstr(
        k_on_2 * (Q_on_2 @ pg2) + k_off_2 * (Q_off_2 @ pg2) + k_tx_2 * (Q_tx_2 @ pg2) + k_deg_2 * (Q_deg_2 @ pg2) == 0,
        name="Marginal_CME_x2"
    )

def add_TE_equality_constraints(model, variables, overall_extent_OG):

    # use extent of threshold truncation for OG states
    max_x1_OG = overall_extent_OG['max_x1_OG']
    max_x2_OG = overall_extent_OG['max_x2_OG']

    # get variables
    p1 = variables['p1']
    p2 = variables['p2']
    pg1 = variables['pg1']
    pg2 = variables['pg2']

    # construct A matrices
    A1 = np.repeat(np.eye(max_x1_OG + 1, dtype=int), repeats=2, axis=1)
    A2 = np.repeat(np.eye(max_x2_OG + 1, dtype=int), repeats=2, axis=1)

    # equate p1 and p2 to pg1 and pg2 sums
    model.addConstr(p1 == A1 @ pg1)
    model.addConstr(p2 == A2 @ pg2)

# ------------------------------------------------
# Moment constraints
# ------------------------------------------------

def add_moment_constraints(model, variables, moment_extent_OG, moments_OB, beta):

    # moment OG truncation for sample i
    max_x1_OG = moment_extent_OG['max_x1_OG']
    max_x2_OG = moment_extent_OG['max_x2_OG']

    # get variables
    E_x1 = variables['E_x1']
    E_x2 = variables['E_x2']

    # slice variables to truncation
    p1_slice = variables['p1'][0: max_x1_OG + 1]
    p2_slice = variables['p2'][0: max_x2_OG + 1]

    # get capture efficiency moments
    E_beta = np.mean(beta)
    E_beta_sq = np.mean(beta**2)

    # expressions for moments (OG)
    expr_E_x1 = gp.quicksum(p1_slice * np.arange(max_x1_OG + 1))
    expr_E_x2 = gp.quicksum(p2_slice * np.arange(max_x2_OG + 1))

    # equality constraints (OG)
    model.addConstr(E_x1 == expr_E_x1, name="E_x1_equality")
    model.addConstr(E_x2 == expr_E_x2, name="E_x2_equality")

    # moment bounds (OB CI)
    model.addConstr(E_x1 <= moments_OB['E_x1'][1] / E_beta, name="E_x1_UB")
    model.addConstr(E_x1 >= moments_OB['E_x1'][0] / E_beta, name="E_x1_LB")
    model.addConstr(E_x2 <= moments_OB['E_x2'][1] / E_beta, name="E_x2_UB")
    model.addConstr(E_x2 >= moments_OB['E_x2'][0] / E_beta, name="E_x2_LB")

    # moment independence constraint
    model.addConstr(E_x1 * E_x2 <= moments_OB['E_x1_x2'][1] / E_beta_sq, name="Indep_UB")
    model.addConstr(E_x1 * E_x2 >= moments_OB['E_x1_x2'][0] / E_beta_sq, name="Indep_LB")

def add_higher_moment_constraints(model, variables, moment_extent_OG, moments_OB, beta):

    # use extent of threshold truncation for OG states
    max_x1_OG = moment_extent_OG['max_x1_OG']
    max_x2_OG = moment_extent_OG['max_x2_OG']

    # slice variables to truncation
    p1_slice = variables['p1'][0: max_x1_OG + 1]
    p2_slice = variables['p2'][0: max_x2_OG + 1]

    # get capture efficiency moments
    E_beta = np.mean(beta)
    E_beta_sq = np.mean(beta**2)

    # expressions for moments (OG)
    expr_E_x1_OG = gp.quicksum(p1_slice * np.arange(max_x1_OG + 1))
    expr_E_x2_OG = gp.quicksum(p2_slice * np.arange(max_x2_OG + 1))
    expr_E_x1_sq_OG = gp.quicksum(p1_slice * np.arange(max_x1_OG + 1)**2)
    expr_E_x2_sq_OG = gp.quicksum(p2_slice * np.arange(max_x2_OG + 1)**2)

    # expressions for moments (OB)
    expr_E_x1_sq_OB = expr_E_x1_sq_OG*E_beta_sq + expr_E_x1_OG*(E_beta - E_beta_sq)
    expr_E_x2_sq_OB = expr_E_x2_sq_OG*E_beta_sq + expr_E_x2_OG*(E_beta - E_beta_sq)

    # moment bounds (OB CI)
    model.addConstr(expr_E_x1_sq_OB <= moments_OB['E_x1_sq'][1], name="E_x1_sq_UB")
    model.addConstr(expr_E_x1_sq_OB >= moments_OB['E_x1_sq'][0], name="E_x1_sq_LB")
    model.addConstr(expr_E_x2_sq_OB <= moments_OB['E_x2_sq'][1], name="E_x2_sq_UB")
    model.addConstr(expr_E_x2_sq_OB >= moments_OB['E_x2_sq'][0], name="E_x2_sq_LB")

def add_dummy_moment_constraints(model, variables, moments_OB, beta):

    # get variables
    E_x1 = variables['E_x1']
    E_x2 = variables['E_x2']

    # get capture efficiency moments
    E_beta = np.mean(beta)
    E_beta_sq = np.mean(beta**2)

    # moment bounds (dummy variables)
    model.addConstr(E_x1 <= moments_OB['E_x1'][1] / E_beta, name="E_x1_UB")
    model.addConstr(E_x1 >= moments_OB['E_x1'][0] / E_beta, name="E_x1_LB")
    model.addConstr(E_x2 <= moments_OB['E_x2'][1] / E_beta, name="E_x2_UB")
    model.addConstr(E_x2 >= moments_OB['E_x2'][0] / E_beta, name="E_x2_LB")

    # moment independence constraint (dummy variables)
    model.addConstr(E_x1 * E_x2 <= moments_OB['E_x1_x2'][1] / E_beta_sq, name="Indep_UB")
    model.addConstr(E_x1 * E_x2 >= moments_OB['E_x1_x2'][0] / E_beta_sq, name="Indep_LB")

# ------------------------------------------------
# Downsampled constraints
# ------------------------------------------------

def add_downsampled_probability_constraints(model, variables, probs_OB, truncation_OB):

    # get OB truncation for sample i (NOTE: only using upper boundary)
    max_x1_OB = truncation_OB['max_x1_OB']
    max_x2_OB = truncation_OB['max_x2_OB']

    # get variables
    pd = variables['pd']

    # CI bounds
    model.addConstr(pd <= probs_OB['bounds'][1, :max_x1_OB + 1, :max_x2_OB + 1], name="pd_UB")
    model.addConstr(pd >= probs_OB['bounds'][0, :max_x1_OB + 1, :max_x2_OB + 1], name="pd_LB")

def add_downsampled_marginal_probability_1_constraints(model, variables, probs_OB, truncationM_OB):

    # get OB truncation for sample i (NOTE: only using upper boundary)
    max_x1_OB = truncationM_OB['maxM_x1_OB']

    # get variables
    pd1 = variables['pd1']

    # CI bounds
    model.addConstr(pd1 <= probs_OB['x1_bounds'][1, :max_x1_OB + 1], name="pd1_UB")
    model.addConstr(pd1 >= probs_OB['x1_bounds'][0, :max_x1_OB + 1], name="pd1_LB")

def add_downsampled_marginal_probability_2_constraints(model, variables, probs_OB, truncationM_OB):
    
    # get OB truncation for sample i (NOTE: only using upper boundary)
    max_x2_OB = truncationM_OB['maxM_x2_OB']

    # get variables
    pd2 = variables['pd2']

    # CI bounds
    model.addConstr(pd2 <= probs_OB['x2_bounds'][1, :max_x2_OB + 1], name="pd2_UB")
    model.addConstr(pd2 >= probs_OB['x2_bounds'][0, :max_x2_OB + 1], name="pd2_LB")

def add_downsampled_CME_constraints(model, variables, fm_OB, truncation_OB):

    # get OB truncation for sample i 
    max_x1_OB = truncation_OB['max_x1_OB']
    max_x2_OB = truncation_OB['max_x2_OB']

    # get variables
    pd = variables['pd']
    fm = variables['fm']
    k_tx_1 = variables['k_tx_1']
    k_tx_2 = variables['k_tx_2']
    k_deg_1 = variables['k_deg_1']
    k_deg_2 = variables['k_deg_2']
    k_reg = variables['k_reg']

    # fm rate bounds
    model.addConstr(fm <= fm_OB['fm1m2'][1, :max_x1_OB + 1, :max_x2_OB + 1], name="fm_UB")
    model.addConstr(fm >= fm_OB['fm1m2'][0, :max_x1_OB + 1, :max_x2_OB + 1], name="fm_LB")

    # dummy zero variable for non-linear constraints
    z = model.addVar()
    model.addConstr(z == 0)
    
    # manually add x1_OB = x2_OB = 0 constraint (to avoid pd1(-1), pd2(-1) terms)
    model.addConstr(
        z == k_deg_1 * pd[1, 0] + \
        k_deg_2 * pd[0, 1] + \
        k_reg * pd[1, 1] - \
        (k_tx_1 * fm[0, 0] + k_tx_2 * fm[0, 0]) * pd[0, 0],
        name="CME_d_0_0"
    )

    # manually add x1_OB = 0 constraints (to avoid pd1(-1) terms)
    model.addConstrs(
        (
            z == k_tx_2 * fm[0, x2_OB - 1] * pd[0, x2_OB - 1] + \
            k_deg_1 * pd[1, x2_OB] + \
            k_deg_2 * (x2_OB + 1) * pd[0, x2_OB + 1] + \
            k_reg * (x2_OB + 1) * pd[1, x2_OB + 1] - \
            (k_tx_1 * fm[0, x2_OB] + k_tx_2 * fm[0, x2_OB] + k_deg_2 * x2_OB) * pd[0, x2_OB]
            for x2_OB in range(1, max_x2_OB)
        ),
        name="CME_d_0_x2"
    )
    # manually add x2_OB = 0 constraints (to avoid pd2(-1) terms)
    model.addConstrs(
        (
            z == k_tx_1 * fm[x1_OB - 1, 0] * pd[x1_OB - 1, 0] + \
            k_deg_1 * (x1_OB + 1) * pd[x1_OB + 1, 0] + \
            k_deg_2 * pd[x1_OB, 1] + \
            k_reg * (x1_OB + 1) * pd[x1_OB + 1, 1] - \
            (k_tx_1 * fm[x1_OB, 0] + k_tx_2 * fm[x1_OB, 0] + k_deg_1 * x1_OB) * pd[x1_OB, 0]
            for x1_OB in range(1, max_x1_OB)
        ),
        name="CME_d_x1_0"
    )

    # add CME constraints
    model.addConstrs(
        (
            z == k_tx_1 * fm[x1_OB - 1, x2_OB] * pd[x1_OB - 1, x2_OB] + \
            k_tx_2 * fm[x1_OB, x2_OB - 1] * pd[x1_OB, x2_OB - 1] + \
            k_deg_1 * (x1_OB + 1) * pd[x1_OB + 1, x2_OB] + \
            k_deg_2 * (x2_OB + 1) * pd[x1_OB, x2_OB + 1] + \
            k_reg * (x1_OB + 1) * (x2_OB + 1) * pd[x1_OB + 1, x2_OB + 1] - \
            (k_tx_1 * fm[x1_OB, x2_OB] + k_tx_2 * fm[x1_OB, x2_OB] + k_deg_1 * x1_OB + k_deg_2 * x2_OB + k_reg * x1_OB * x2_OB) * pd[x1_OB, x2_OB]
            for x1_OB in range(1, max_x1_OB)
            for x2_OB in range(1, max_x2_OB)
        ),
        name="CME_d_x1_x2"
    )

def add_downsampled_marginal_CME_1_constraints(model, variables, fm_OB, truncationM_OB):
    
    # get OB truncation for sample i 
    max_x1_OB = truncationM_OB['maxM_x1_OB']

    # get variables
    pd1 = variables['pd1']
    fm1 = variables['fm1']
    k_tx_1 = variables['k_tx_1']
    k_deg_1 = variables['k_deg_1']

    # fm rate bounds
    model.addConstr(fm1 <= fm_OB['fm1'][1, :max_x1_OB + 1], name="fm1_UB")
    model.addConstr(fm1 >= fm_OB['fm1'][0, :max_x1_OB + 1], name="fm1_LB")

    # dummy zero variable for non-linear constraints
    z = model.addVar()
    model.addConstr(z == 0)

    # manually add x1_OB = 0 constraint (to avoid pd1(-1))
    model.addConstr(
        z == k_deg_1 * pd1[1] - k_tx_1 * fm1[0] * pd1[0],
        name="CME_d_x1_0"
    )

    # x1_OB CME
    model.addConstrs(
        (
            z == k_tx_1 * fm1[x1_OB - 1] * pd1[x1_OB - 1] + \
            k_deg_1 * (x1_OB + 1) * pd1[x1_OB + 1] - \
            (k_tx_1 * fm1[x1_OB] + k_deg_1 * x1_OB) * pd1[x1_OB]
            for x1_OB in range(1, max_x1_OB)
        ),
        name="CME_d_x1"
    )

def add_downsampled_marginal_CME_2_constraints(model, variables, fm_OB, truncationM_OB):
    
    # get OB truncation for sample i
    max_x2_OB = truncationM_OB['maxM_x2_OB']

    # get variables
    pd2 = variables['pd2']
    fm2 = variables['fm2']
    k_tx_2 = variables['k_tx_2']
    k_deg_2 = variables['k_deg_2']

    # fm rate bounds
    model.addConstr(fm2 <= fm_OB['fm2'][1, :max_x2_OB + 1], name="fm2_UB")
    model.addConstr(fm2 >= fm_OB['fm2'][0, :max_x2_OB + 1], name="fm2_LB")

    # dummy zero variable for non-linear constraints
    z = model.addVar()
    model.addConstr(z == 0)

    # manually add x2_OB = 0 constraint (to avoid pd2(-1))
    model.addConstr(
        z == k_deg_2 * pd2[1] - k_tx_2 * fm2[0] * pd2[0],
        name="CME_d_x2_0"
    )

    # x2_OB CME
    model.addConstrs(
        (
            z == k_tx_2 * fm2[x2_OB - 1] * pd2[x2_OB - 1] + \
            k_deg_2 * (x2_OB + 1) * pd2[x2_OB + 1] - \
            (k_tx_2 * fm2[x2_OB] + k_deg_2 * x2_OB) * pd2[x2_OB]
            for x2_OB in range(1, max_x2_OB)
        ),
        name="CME_d_x1"
    )
