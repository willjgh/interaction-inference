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
# Constraints
# ------------------------------------------------

def add_joint_probability_constraints(model, variables, sample_truncation_OB, truncation_OG, i):

    # get OB truncation for sample i
    min_x1_OB = sample_truncation_OB['min_x1_OB']
    max_x1_OB = sample_truncation_OB['max_x1_OB']
    min_x2_OB = sample_truncation_OB['min_x2_OB']
    max_x2_OB = sample_truncation_OB['max_x2_OB']

    # load CI bounds for sample i
    bounds = np.load(f"./Test-Info/Bounds/Joint/sample-{i}.npy")
            
    # for each OB state pair in truncation
    for x1_OB in range(min_x1_OB, max_x1_OB + 1):
        for x2_OB in range(min_x2_OB, max_x2_OB + 1):

            # get OG truncation for OB state pair
            min_x1_OG, max_x1_OG = truncation_OG[x1_OB]
            min_x2_OG, max_x2_OG = truncation_OG[x2_OB]
            
            # load coefficient grid for OB state pair
            B_coeffs = np.load(f"./Test-Info/Coefficients/state-{x1_OB}-{x2_OB}.npy")

            # slice variables to truncation
            p1_slice = variables['p1'][min_x1_OG: max_x1_OG + 1]
            p2_slice = variables['p2'][min_x2_OG: max_x2_OG + 1]

            # bilinear form
            sum_expr = p1_slice.T @ B_coeffs @ p2_slice
        
            # form constraints using CI bounds
            model.addConstr(sum_expr >= bounds[0, x1_OB, x2_OB], name=f"B_lb_{x1_OB}_{x2_OB}")
            model.addConstr(sum_expr <= bounds[1, x1_OB, x2_OB], name=f"B_ub_{x1_OB}_{x2_OB}")

def add_marginal_probability_constraints(model, variables, sample_truncationM_OB, truncation_OG, i):

    # get marginal OB truncation for sample i
    minM_x1_OB = sample_truncationM_OB['minM_x1_OB']
    maxM_x1_OB = sample_truncationM_OB['maxM_x1_OB']
    minM_x2_OB = sample_truncationM_OB['minM_x2_OB']
    maxM_x2_OB = sample_truncationM_OB['maxM_x2_OB']

    # load CI bounds for sample i
    x1_bounds = np.load(f"./Test-Info/Bounds/x1_marginal/sample-{i}.npy")
    x2_bounds = np.load(f"./Test-Info/Bounds/x2_marginal/sample-{i}.npy")

    # for each OB state in truncation
    for x1_OB in range(minM_x1_OB, maxM_x1_OB + 1):

        # get OG truncation
        min_x1_OG, max_x1_OG = truncation_OG[x1_OB]

        # load marginal coefficient array for OB state
        Bm_coeffs = np.load(f"./Test-Info/Coefficients/state-{x1_OB}.npy")

        # slice variable to truncation
        p1_slice = variables['p1'][min_x1_OG: max_x1_OG + 1]

        # linear expression of sum
        sum_expr = gp.quicksum(Bm_coeffs * p1_slice)

        # form constraints using CI bounds
        model.addConstr(sum_expr >= x1_bounds[0, x1_OB], name=f"Bm_x1_lb_{x1_OB}")
        model.addConstr(sum_expr <= x1_bounds[1, x1_OB], name=f"Bm_x1_ub_{x1_OB}")

    # repeat for x2
    for x2_OB in range(minM_x2_OB, maxM_x2_OB + 1):

        # get OG truncation
        min_x2_OG, max_x2_OG = truncation_OG[x2_OB]

        # load marginal coefficient array for OB state
        Bm_coeffs = np.load(f"./Test-Info/Coefficients/state-{x2_OB}.npy")

        # slice variable to truncation
        p2_slice = variables['p2'][min_x2_OG: max_x2_OG + 1]

        # linear expression of sum
        sum_expr = gp.quicksum(Bm_coeffs * p2_slice)

        # form constraints using CI bounds
        model.addConstr(sum_expr >= x2_bounds[0, x2_OB], name=f"Bm_x2_lb_{x2_OB}")
        model.addConstr(sum_expr <= x2_bounds[1, x2_OB], name=f"Bm_x2_ub_{x2_OB}")

def add_moment_constraints(model, variables, sample_moments_OB, sample_moment_extent_OG, beta):

    # moment OG truncation for sample i
    max_x1_OG = sample_moment_extent_OG['max_x1_OG']
    max_x2_OG = sample_moment_extent_OG['max_x2_OG']

    # get variables
    p1 = variables['p1']
    p2 = variables['p2']

    # get capture efficiency moments
    E_beta = np.mean(beta)
    E_beta_sq = np.mean(beta**2)

    # expressions for moments (OG)
    expr_E_x1 = gp.quicksum(p1 * np.arange(max_x1_OG + 1))
    expr_E_x2 = gp.quicksum(p2 * np.arange(max_x2_OG + 1))

    # moment bounds (OB CI)
    model.addConstr(expr_E_x1 <= sample_moments_OB['E_x1'][1] / E_beta, name="E_x1_UB")
    model.addConstr(expr_E_x1 >= sample_moments_OB['E_x1'][0] / E_beta, name="E_x1_LB")
    model.addConstr(expr_E_x2 <= sample_moments_OB['E_x2'][1] / E_beta, name="E_x2_UB")
    model.addConstr(expr_E_x2 >= sample_moments_OB['E_x2'][0] / E_beta, name="E_x2_LB")

    # moment independence constraint
    model.addConstr(expr_E_x1 * expr_E_x2 <= sample_moments_OB['E_x1_x2'][1] / E_beta_sq, name="Indep_UB")
    model.addConstr(expr_E_x1 * expr_E_x2 >= sample_moments_OB['E_x1_x2'][0] / E_beta_sq, name="Indep_LB")

def add_higher_moment_constraints(model, variables, sample_moment_extent_OG, sample_moments_OB, beta):

    # use extent of threshold truncation for OG states
    max_x1_OG = sample_moment_extent_OG['max_x1_OG']
    max_x2_OG = sample_moment_extent_OG['max_x2_OG']

    # get variables
    p1 = variables['p1']
    p2 = variables['p2']

    # get capture efficiency moments
    E_beta = np.mean(beta)
    E_beta_sq = np.mean(beta**2)

    # expressions for moments (OG)
    expr_E_x1_OG = gp.quicksum(p1 * np.arange(max_x1_OG + 1))
    expr_E_x2_OG = gp.quicksum(p2 * np.arange(max_x2_OG + 1))
    expr_E_x1_sq_OG = gp.quicksum(p1 * np.arange(max_x1_OG + 1)**2)
    expr_E_x2_sq_OG = gp.quicksum(p2 * np.arange(max_x2_OG + 1)**2)

    # expressions for moments (OB)
    expr_E_x1_sq_OB = expr_E_x1_sq_OG*E_beta_sq + expr_E_x1_OG*(E_beta - E_beta_sq)
    expr_E_x2_sq_OB = expr_E_x2_sq_OG*E_beta_sq + expr_E_x2_OG*(E_beta - E_beta_sq)

    # moment bounds (OB CI)
    model.addConstr(expr_E_x1_sq_OB <= sample_moments_OB['E_x1_sq'][1], name="E_x1_sq_UB")
    model.addConstr(expr_E_x1_sq_OB >= sample_moments_OB['E_x1_sq'][0], name="E_x1_sq_LB")
    model.addConstr(expr_E_x2_sq_OB <= sample_moments_OB['E_x2_sq'][1], name="E_x2_sq_UB")
    model.addConstr(expr_E_x2_sq_OB >= sample_moments_OB['E_x2_sq'][0], name="E_x2_sq_LB")

def add_CME_constraints(model, variables, sample_overall_extent_OG):

    # get extent of OG states
    max_x1_OG = sample_overall_extent_OG['max_x1_OG']
    max_x2_OG = sample_overall_extent_OG['max_x2_OG']

    # get variables
    p = variables['p']
    k_tx_1 = variables['k_tx_1']
    k_tx_2 = variables['k_tx_2']
    k_deg_1 = variables['k_deg_2']
    k_deg_2 = variables['k_deg_1']
    
    # manually add x1_OG = x2_OG = 0 constraint (to avoid p(0) terms)
    model.addConstr(
        0 == k_deg_1 * p[1, 0] + \
        k_deg_2 * p[0, 1] - \
        (k_tx_1 + k_tx_2) * p[0, 0],
        name="CME_0_0"
    )

    # manually add x1_OG = 0 constraints (to avoid p1(-1) terms)
    model.addConstrs(
        (
            0 == k_tx_2 * p[0, x2_OG - 1] + \
            k_deg_1 * p[1, x2_OG] + \
            k_deg_2 * (x2_OG + 1) * p[0, x2_OG + 1] - \
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
            k_deg_2 * p[x1_OG, 1] - \
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
            k_deg_2 * (x2_OG + 1) * p[x1_OG, x2_OG + 1] - \
            (k_tx_1 + k_tx_2 + k_deg_1 * x1_OG + k_deg_2 * x2_OG) * p[x1_OG, x2_OG]
            for x1_OG in range(1, max_x1_OG)
            for x2_OG in range(1, max_x2_OG)
        ),
        name="CME_x1_x2"
    )

def add_marginal_CME_constraints(model, variables, sample_overall_extent_OG):

    # get extent of OG states
    max_x1_OG = sample_overall_extent_OG['max_x1_OG']
    max_x2_OG = sample_overall_extent_OG['max_x2_OG']

    # get variables
    p1 = variables['p1']
    p2 = variables['p2']
    k_tx_1 = variables['k_tx_1']
    k_tx_2 = variables['k_tx_2']
    k_deg_1 = variables['k_deg_2']
    k_deg_2 = variables['k_deg_1']

    # construct Q matrices: 1 more column than square to add upper diagonal to last row
    Q_tx_1 = (np.diag([1 for x in range(1, max_x1_OG + 1)], -1) - np.diag([1 for x in range(max_x1_OG + 1)]))[:-1, :]
    Q_tx_2 = (np.diag([1 for x in range(1, max_x2_OG + 1)], -1) - np.diag([1 for x in range(max_x2_OG + 1)]))[:-1, :]
    Q_deg_1 = (np.diag([x for x in range(1, max_x1_OG + 1)], 1) - np.diag([x for x in range(max_x1_OG + 1)]))[:-1, :]
    Q_deg_2 = (np.diag([x for x in range(1, max_x2_OG + 1)], 1) - np.diag([x for x in range(max_x2_OG + 1)]))[:-1, :]

    # add matrix constraints
    model.addConstr(
        k_tx_1 * (Q_tx_1 @ p1) + k_deg_1 * (Q_deg_1 @ p1) == 0,
        name="Marginal_CME_x1"
    )

    model.addConstr(
        k_tx_2 * (Q_tx_2 @ p2) + k_deg_2 * (Q_deg_2 @ p2) == 0,
        name="Marginal_CME_x2"
    )

def add_base_constraints(model, variables):

    # fix k_deg_1 = 1, k_deg = 2 for identifiability
    model.addConstr(variables['k_deg_1'] == 1, name="Fix_k_deg_1")
    model.addConstr(variables['k_deg_2'] == 1, name="Fix_k_deg_2")

    # distributional constraints
    model.addConstr(variables['p1'].sum() <= 1, name="Dist_x1")
    model.addConstr(variables['p2'].sum() <= 1, name="Dist_x2")

def add_factorization_constraints(model, variables):

    # get variables
    p1 = variables['p1']
    p2 = variables['p2']
    p = variables['p']

    # outer product marginals
    outer = p1[:, None] @ p2[None, :]

    # equate dummy joint variable to product of marginals: all original states
    model.addConstr(p == outer, name=f"Joint_factorize")