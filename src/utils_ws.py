"""
utils.py

This module provides utility functions for numerical operations on numpy arrays.
It includes functions to calculate the mean of an array and to normalize an array
so that its values range between 0 and 1.

Dependencies:
- numpy: This module relies on numpy for efficient numerical computations.

Functions:
- calculate_mean(arr): Computes the mean of a numpy array.
- normalize_array(arr): Normalizes a numpy array to a 0-1 range.
"""

import numpy as np

def calculate_mean(arr):
    """
    Calculate the mean of a numpy array.

    Parameters:
    - arr (numpy.ndarray): A numpy array containing numerical values.

    Returns:
    - float: The mean value of the array elements.

    Example:
    >>> calculate_mean(np.array([1, 2, 3, 4]))
    2.5
    """
    return np.mean(arr)  # Uses numpy's mean function for efficient computation.

def normalize_array(arr):
    """
    Normalize a numpy array so its values range between 0 and 1.

    Parameters:
    - arr (numpy.ndarray): A numpy array containing numerical values.

    Returns:
    - numpy.ndarray: A new array with values normalized to the range [0, 1].

    Example:
    >>> normalize_array(np.array([1, 2, 3, 4]))
    array([0. , 0.33333333, 0.66666667, 1. ])

    Note:
    - If all elements in the array are the same, the function returns the original array
      to avoid division by zero.
    """
    min_val = np.min(arr)  # Find the minimum value in the array.
    max_val = np.max(arr)  # Find the maximum value in the array.
    # Normalize the array only if max_val is not equal to min_val to avoid division by zero.
    return (arr - min_val) / (max_val - min_val) if max_val != min_val else arr