#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
A function to get the most frequent (mode) value from a list
"""


def get_mode(list_of_values):
    
    import numpy as np
    
    # find all unique values and recreate the list with the indices of those
    # unique values
    unique_values, list_of_indices = np.unique(list_of_values,
                                             return_inverse=True)
    
    # count the number of each index value
    index_counts = np.bincount(list_of_indices)
    
    # find the most frequent index (i.e., largest count)
    most_frequent_index = index_counts.argmax()
    
    # get the unique value from the original list with the correspoinding index
    most_frequent_value = unique_values[most_frequent_index]
    
    return most_frequent_value