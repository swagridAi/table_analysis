"""
Functions for calculating co-occurrence between data elements.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict

def calculate_cooccurrence(df_exploded):
    """
    Calculate co-occurrence of data elements across reports.
    
    Args:
        df_exploded (pandas.DataFrame): Exploded dataframe with 'Report' and 'data_element' columns
        
    Returns:
        dict: Dictionary with (element1, element2) tuples as keys and co-occurrence counts as values
    """
    # Group by Report to get all data elements used in each report
    report_groups = df_exploded.groupby("Report")["data_element"].apply(list).reset_index()
    
    # Build co-occurrence dictionary
    co_occurrence = defaultdict(int)
    
    for row in report_groups.itertuples():
        elements_used = getattr(row, "data_element")
        # Use set to ensure unique elements only
        for combo in combinations(sorted(set(elements_used)), 2):
            co_occurrence[combo] += 1
    
    return co_occurrence

def create_cooccurrence_matrix(co_occurrence):
    """
    Convert co-occurrence dictionary to a matrix.
    
    Args:
        co_occurrence (dict): Dictionary with (element1, element2) tuples as keys 
                              and co-occurrence counts as values
        
    Returns:
        tuple: (pandas.DataFrame, list) - Co-occurrence matrix as DataFrame and list of all elements
    """
    # Get all unique data elements
    all_elements = set()
    for pair in co_occurrence.keys():
        all_elements.add(pair[0])
        all_elements.add(pair[1])
    
    all_elements = sorted(list(all_elements))
    
    # Create empty matrix
    matrix_size = len(all_elements)
    matrix = np.zeros((matrix_size, matrix_size))
    
    # Create mapping from element names to indices
    element_to_idx = {element: i for i, element in enumerate(all_elements)}
    
    # Fill the matrix with co-occurrence counts
    for (elem1, elem2), count in co_occurrence.items():
        i = element_to_idx[elem1]
        j = element_to_idx[elem2]
        matrix[i, j] = count
        matrix[j, i] = count  # Mirror the matrix since co-occurrence is symmetric
    
    # Create a DataFrame for better readability
    matrix_df = pd.DataFrame(matrix, index=all_elements, columns=all_elements)
    
    return matrix_df, all_elements