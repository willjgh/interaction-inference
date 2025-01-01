'''
Module to implement functions for analysis of performance on results on datsets.
'''

# ------------------------------------------------
# Dependencies
# ------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

# ------------------------------------------------
# Plotting Functions
# ------------------------------------------------

def scatter_parameters(dataset):
    '''
    Produce scatter plot of parameters of the dataset.
    '''

    # exit if no paramters available
    if dataset.param_dataset is None:
        print("No parameter dataset available")
        return None

    # get parameter dataset
    params_df = dataset.param_dataset

    # setup plotting
    fig, axs = plt.subplots(1, figsize=(12, 12))

    sc = axs.scatter(
        np.log10(params_df['k_tx_1'].astype(np.float64)) - np.log10(params_df['k_deg_1'].astype(np.float64)),
        np.log10(params_df['k_tx_2'].astype(np.float64)) - np.log10(params_df['k_deg_2'].astype(np.float64)),
        c=np.log10(params_df['k_reg'].astype(np.float64)),
        label="Colour = log(k_reg)",
        cmap=plt.cm.viridis
    )

    plt.colorbar(sc)
    axs.set_xlabel("log(k_tx_1 / k_deg_1)")
    axs.set_ylabel("log(k_tx_2 / k_deg_2)")
    axs.set_title(f"Distribution of parameters in dataset")
    axs.legend()
    plt.show()

def scatter_results(dataset, result):
    '''
    Scatter plot gene-pair parameters, coloured by correct / false detection.
    '''

    # exit if no paramters available
    if dataset.param_dataset is None:
        print("No parameter dataset available")
        return None

    # get parameter dataset
    params_df = dataset.param_dataset

    # set figure
    if (result.method == "hyp") or (result.method == "min"):
        # add time plot for optimization methods
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        time_data = []
    else:
        # only scatter plot for correlation tests
        fig, axs = plt.subplots(1, 1, figsize=(12, 12))

    # flags for legend
    green_label_needed = True
    red_label_needed = True
    blue_label_needed = True

    # loop over each gene-pair
    for key, val in result.result_dict.items():

        # reset label
        label = None

        # true parameters
        params = params_df.loc[f'Gene-pair-{key}']

        # set true status of interaction
        if params['k_reg'] > 0:
            interaction = True
        else:
            interaction = False

        # extract time
        if (result.method == "hyp") or (result.method == "min"):
            time_data.append(val['time'])

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
            if detected == True:
                color = "green"
                if green_label_needed:
                    label = "Interaction detected"
                    green_label_needed = False
            elif detected == False:
                color = "red"
                if red_label_needed:
                    label = "Interaction not detected"
                    red_label_needed = False
            else:
                color = "blue"
                if blue_label_needed:
                    if (result.method == "spearman") or (result.method == "pearson"):
                        label="Undefined"
                    else:
                        label="Time limit"
                    blue_label_needed = False

        # set colour according to detection of interaction: data has no interaction
        else:
            if detected == False:
                color = "green"
                if green_label_needed:
                    label = "True negative"
                    green_label_needed = False
            elif detected == True:
                color = "red"
                if red_label_needed:
                    label = "False positive"
                    red_label_needed = False
            else:
                color = "blue"
                if blue_label_needed:
                    if (result.method == "spearman") or (result.method == "pearson"):
                        label = "Undefined"
                    else:
                        label="Time limit"
                    blue_label_needed = False

        # plot point: (x, y) location as log-mean of genes, colour as detection
        axs[0].scatter(
            x = np.log10(params['k_tx_1']) - np.log10(params['k_deg_1']),
            y = np.log10(params['k_tx_2']) - np.log10(params['k_deg_2']),
            color = color,
            label = label
        )

    # format parameter scatter
    axs[0].set_xlabel("log(k_tx_1 / k_deg_1)")
    axs[0].set_ylabel("log(k_tx_2 / k_deg_2)")
    axs[0].set_title(f"Distribution of parameters and detection results for {result.method} method")
    axs[0].legend()

    # format time histogram
    if (result.method == "hyp") or (result.method == "min"):
        total_time = int(sum(time_data))
        total_time_str = f"{total_time // 3600} h {total_time // 60} m {total_time % 60}"
        axs[1].hist(time_data, label=f"Total time: " + total_time_str)
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Frequency")
        axs[1].set_title("Optimization time")
        axs[1].legend()

    # display
    plt.show()

# ------------------------------------------------
# Classification Functions
# ------------------------------------------------

def compute_confusion_matrix(dataset, result):
    '''
    Produce confusion matrix dictionary from a result dictionary and true params.
    '''

    # exit if no paramters available
    if dataset.param_dataset is None:
        print("No parameter dataset available")
        return None

    # get parameter dataset
    params_df = dataset.param_dataset

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

def classification_performance(dataset_list, result_list):
    '''
    Produce and display confusion matrix and related classification metrics
    given a range of results and true paramters
    '''

    # store overall confusion matrix
    confusion_matrix_overall = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    # iterate over pairs of results and true paramters
    for i, result in enumerate(result_list):
        dataset = dataset_list[i]

        # compute confusion matrix
        confusion_matrix = compute_confusion_matrix(dataset, result)

        # add to overall
        confusion_matrix_overall = add_confusion_matrices(confusion_matrix, confusion_matrix_overall)

    # dispay final confusion matrix
    return display_metrics(confusion_matrix_overall)
