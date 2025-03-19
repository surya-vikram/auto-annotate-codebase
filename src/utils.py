import numpy as np


def calculate_mean(arr):
    """
    Calculate the mean of a numpy array.
    """
    return np.mean(arr)


def normalize_array(arr):
    """
    Normalize a numpy array so its values range between 0 and 1.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val) if max_val != min_val else arr
