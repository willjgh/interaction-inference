'''
Module to implement functions for analysis of performance on results on datsets.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import math

# ------------------------------------------------
# Plotting Functions
# ------------------------------------------------

def scatter_parameters(dataset):
    '''
    Produce a scatter plot of parameters of a dataset.

    For a simulated dataset with true parameters available produce a scatter
    plot of the log mean expression of each gene pair when ignoring interaction
    effects i.e. points (log(k_tx_1 / k_deg_1), log(k_tx_2 / k_deg_2)).
    Gene pairs with interaction (k_reg > 0) are coloured by log interaction
    strength with darker colours for smaller values, and those with no
    interaction are coloured black.

    Args:
        dataset: instance of Dataset class whose 'param_dataset' attribute
                 should not be None
    '''

    # exit if no paramters available
    if dataset.param_dataset is None:
        print("No parameter dataset available")
        return None

    # get parameter dataset
    params_df = dataset.param_dataset

    # setup plotting
    fig, axs = plt.subplots()#1, figsize=(12, 12))

    # plot gene pairs with interaction (k_reg > 0)
    params_interaction = params_df[params_df['k_reg'] > 0]

    sc_int = axs.scatter(
        np.log10(params_interaction['k_tx_1'].astype(np.float64)) - np.log10(params_interaction['k_deg_1'].astype(np.float64)),
        np.log10(params_interaction['k_tx_2'].astype(np.float64)) - np.log10(params_interaction['k_deg_2'].astype(np.float64)),
        c=np.log10(params_interaction['k_reg'].astype(np.float64)),
        label="Interacting gene pairs",
        cmap=plt.cm.plasma
    )

    # plot gene pairs with no interaction (k_reg = 0)
    params_independent = params_df[params_df['k_reg'] == 0]

    sc_ind = axs.scatter(
        np.log10(params_independent['k_tx_1'].astype(np.float64)) - np.log10(params_independent['k_deg_1'].astype(np.float64)),
        np.log10(params_independent['k_tx_2'].astype(np.float64)) - np.log10(params_independent['k_deg_2'].astype(np.float64)),
        c='black',
        label="Independent gene pairs"
    )

    plt.colorbar(sc_int, label="(log10) Interaction strength")
    axs.set_xlabel("(log10) Mean expression of gene 1")
    axs.set_ylabel("(log10) Mean expression of gene 2")
    axs.set_title(f"Scatter plot of dataset parameters")
    axs.legend()
    plt.show()

def scatter_results(result, detailed=False):
    '''
    Produce a scatter plot of interaction detection performance.

    For a simulated dataset with true parameters available produce a scatter
    plot of the log mean expression of each gene pair when ignoring interaction
    effects i.e. points (log(k_tx_1 / k_deg_1), log(k_tx_2 / k_deg_2)) with
    points coloured by classification results: green and red for correct and
    incorrect classification, or specific colours for true positives, false
    positives, true negatives and false negatives when 'detailed' bool True. 

    Args:
        result: instance of Hypothesis, Minimization or Correlation class which
                was used to analyse a dataset whose 'param_dataset' attribute
                should not be None (i.e. simulated dataset)
        detailed: bool for colour detail of scatter plot, False colours only 
                  correct and incorrect classification whereas True colours each
                  of True / False positive / negative results.
    '''

    # exit if no paramters available
    if result.dataset.param_dataset is None:
        print("No parameter dataset available")
        return None

    # get parameter dataset
    params_df = result.dataset.param_dataset

    # set figure
    fig, axs = plt.subplots()

    # set colours
    if detailed:
        TP_col = "green"
        FN_col = "orange"
        TN_col = "blue"
        FP_col = "red"
        error_col = "black"
    else:
        TP_col = "green"
        FN_col = "red"
        TN_col = "green"
        FP_col = "red"
        error_col = "black"

    # loop over each gene-pair
    for key, val in result.result_dict.items():

        # true parameters
        params = params_df.loc[f'Gene-pair-{key}']

        # set true status of interaction
        if params['k_reg'] > 0:
            interaction = True
        else:
            interaction = False

        # decide if interaction was detected
        if result.method == "hyp":
            if val['status'] == 'INFEASIBLE':
                detected = True
            elif val['status'] == 'OPTIMAL':
                detected = False
            else:
                detected = None
        elif result.method == "min":
            if val['status'] == 'USER_OBJ_LIMIT':
                detected = True
            elif (val['status'] == 'OPTIMAL') and (val['bound'] > 0.0001) and (val['bound'] < np.inf):
                detected = True
            elif val['status'] == 'OPTIMAL':
                detected = False
            else:
                detected = None
        elif (result.method == "pearson") or (result.method == "spearman"):
            if val['pvalue'] < 0.05:
                detected = True
            elif val['pvalue'] >= 0.05:
                detected = False
            else:
                detected = None

        # set colour according to detection of interaction: data has interaction
        if interaction:
            # True positive
            if detected == True:
                color = TP_col
            # False negative
            elif detected == False:
                color = FN_col
            # Error
            else:
                color = error_col

        # set colour according to detection of interaction: data has no interaction
        else:
            # True negative
            if detected == False:
                color = TN_col
            # False positive
            elif detected == True:
                color = FP_col
            # Error
            else:
                color = error_col

        # plot point: (x, y) location as log-mean of genes, colour as detection
        axs.scatter(
            x = np.log10(params['k_tx_1']) - np.log10(params['k_deg_1']),
            y = np.log10(params['k_tx_2']) - np.log10(params['k_deg_2']),
            color = color
        )

    # format parameter scatter
    axs.set_xlabel("(log10) Mean expression of gene 1")
    axs.set_ylabel("(log10) Mean expression of gene 2")
    axs.set_title(f"Scatter plot of dataset parameters and interaction detection")

    # legend formatting
    if detailed:
        handles = [
            Line2D([0], [0], c=TP_col, marker='o', linestyle=''),
            Line2D([0], [0], c=FN_col, marker='o', linestyle=''),
            Line2D([0], [0], c=TN_col, marker='o', linestyle=''),
            Line2D([0], [0], c=FP_col, marker='o', linestyle=''),
            Line2D([0], [0], c=error_col, marker='o', linestyle=''),
        ]
    else:
        handles = [
            Line2D([0], [0], c=TP_col, marker='o', linestyle=''),
            Line2D([0], [0], c=FP_col, marker='o', linestyle=''),
            Line2D([0], [0], c=error_col, marker='o', linestyle='')
        ]

    if (result.method == "hyp" or result.method == "min"):
        error_str = "Time limit"
    else:
        error_str = "Undefined"

    if detailed:
        labels = [
            "True positive",
            "False negative",
            "True negative",
            "False positive",
            error_str
        ]
    else:
        labels = [
            "Correctly classified",
            "Incorrectly classified",
            error_str
        ]
    axs.legend(handles, labels)

    # display
    plt.show()

def time_histogram(result):
    '''
    Display histogram of optimization time per sample when analysing dataset.
    '''

    # check time present
    if not (result.method == "hyp" or result.method == "min"):
        print("No time information available")
        return None

    # extract time data
    time_data = []
    for val in result.result_dict.values():
        time_data.append(val['time'])

    # get total time
    total_time = int(sum(time_data))
    total_time_str = f"{total_time // 3600} h {total_time // 60} m {total_time % 60}"
    
    # time histogram
    fig, axs = plt.subplots()
    axs.hist(time_data, label="Total time: " + total_time_str)
    axs.set_xlabel("Optimizaton time (s)")
    axs.set_ylabel("Frequency")
    axs.set_title("Histogram of optimization time per gene pair")
    axs.legend()

# ------------------------------------------------
# Classification Functions
# ------------------------------------------------

def compute_confusion_matrix(result):
    '''
    Produce confusion matrix dictionary from results.
    '''

    # exit if no paramters available
    if result.dataset.param_dataset is None:
        print("No parameter dataset available")
        return None

    # get parameter dataset
    params_df = result.dataset.param_dataset

    # dataset size
    n = len(result.result_dict)

    # store classification results
    confusion_matrix = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    # loop over gene pairs
    for key, val in result.result_dict.items():

        # access true parameters
        params = params_df.loc[f'Gene-pair-{key}']

        # truth
        if (params['k_reg'] > 0):
            # positive (interaction)
            truth = 1
        else:
            # negative (no interaction)
            truth = 0

        # prediction
        if result.method == "hyp":
            if val['status'] == 'INFEASIBLE':
                # positive predicted (interaction detected)
                pred = 1
            else:
                # negative predicted (interaction not detected) [include time limit, etc.]
                pred = 0

        elif result.method == "min":
            # positive predicted (interaction detected)
            if val['status'] == 'USER_OBJ_LIMIT':
                pred = 1
            elif (val['status']== 'OPTIMAL') and (val['bound'] > 0.0001) and (val['bound'] < np.inf):
                pred = 1
            # negative predicted (interaction not detected) [include infeasible, time limit, etc.]
            else:
                pred = 0
                
        else:
            if val['pvalue'] < 0.05:
                # positive predicted (interaction detected)
                pred = 1
            else:
                # negative predicted (interaction not detected) [include nan, etc.]
                pred = 0

        
        # store result
        if truth == 1:
            if pred == 1:
                confusion_matrix['TP'] += 1 / n
            else:
                confusion_matrix['FN'] += 1 / n
        else:
            if pred == 1:
                confusion_matrix['FP'] += 1 / n
            else:
                confusion_matrix['TN'] += 1 / n

    return confusion_matrix

def get_size(confusion_matrix):
    '''Get dataset size from confusion matrix'''
    
    # store size
    n = 0

    # sum up all result classes
    for key in ['TP', 'FP', 'FN', 'TN']:
        n += confusion_matrix[key]

    return n

def add_confusion_matrices(confusion_matrix_1, confusion_matrix_2):
    '''Combine 2 confusion matrices'''

    # get dataset sizes
    n_1 = get_size(confusion_matrix_1)
    n_2 = get_size(confusion_matrix_2)

    # new confusion matrix
    confusion_matrix = {}

    # combine
    for key in ['TP', 'FP', 'FN', 'TN']:

        confusion_matrix[key] = (n_1 * confusion_matrix_1[key] + n_2 * confusion_matrix_2[key]) / (n_1 + n_2)

    return confusion_matrix

def display_metrics(confusion_matrix):
    '''Display confusion matrix and related classification metrics.'''

    # compute measures
    try:
        accuracy = (confusion_matrix['TP'] + confusion_matrix['TN']) / (confusion_matrix['TP'] + confusion_matrix['TN'] + confusion_matrix['FP'] + confusion_matrix['FN'])
        accuracy = round(accuracy, 2)
    except:
        accuracy = "error"
    try:
        precision = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])
        precision = round(precision, 2)
    except:
        precision = "error"
    try:
        recall = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])
        recall = round(recall, 2)
    except:
        recall = "error"
    try:
        F1_score = 2 * (precision * recall) / (precision + recall)
        F1_score = round(F1_score, 2)
    except:
        F1_score = "error"

    # print measures
    print(f"accuracy = {accuracy}")
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    print(f"F1 score = {F1_score}")

    # display confusion matrix
    values = [[confusion_matrix['TP'], confusion_matrix['FP']], [confusion_matrix['FN'], confusion_matrix['TN']]]
    confusion_matrix_df = pd.DataFrame(data=values, index=['Predicted Positive', 'Predicted Negative'], columns=['Positive', 'Negative'])
    cm = sns.light_palette("blue", as_cmap=True)
    return confusion_matrix_df.style.format(precision=2).background_gradient(cmap=cm, axis=None)

def classification_performance(*result_tuple):
    '''
    Produce and display confusion matrix and related classification metrics
    given result(s): multiple results will be combined into one
    '''

    # store overall confusion matrix
    confusion_matrix_overall = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    # iterate over results
    for result in result_tuple:

        # compute confusion matrix
        confusion_matrix = compute_confusion_matrix(result)

        # add to overall
        confusion_matrix_overall = add_confusion_matrices(confusion_matrix, confusion_matrix_overall)

    # dispay final confusion matrix
    return display_metrics(confusion_matrix_overall)
