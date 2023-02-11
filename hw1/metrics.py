import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for pred_val, true_val in zip(y_pred, y_true):
        if pred_val == true_val == '1': tp += 1
        if pred_val == '1' and true_val == '0': fp += 1
        if pred_val == true_val == '0': tn += 1
        if pred_val == '0' and true_val == '1': fp += 1

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError as err:
         print(f'{err}: zero sum of true positive and false positive; precision is being set to 0.0')
         precision = 0.0
    
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError as err:
         print(f'{err}: zero sum of true positive and false negative; recall is being set to 0.0')
         recall = 0.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError as err:
         print(f'{err}: zero sum of precision and recall; f1 is being set to 0.0')
         f1 = 0.0

    accuracy = (tp + tn) / y_true.shape[0]

    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    accurate_pred = 0

    for pred_val, true_val in zip(y_pred, y_true):
        if pred_val == true_val: accurate_pred += 1
    accuracy = accurate_pred / y_true.shape[0]

    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    ss_red = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - np.average(y_true)))
    r2 = 1 - ss_red / ss_tot

    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = np.sum(np.square(y_true - y_pred)) / y_true.shape[0]

    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = np.sum(np.abs(y_true - y_pred)) / y_true.shape[0]

    return mae
    
