"""
Metrics for evaluating product groups based on co-occurrence data.

This module provides functions to calculate metrics that assess the effectiveness
of potential product groupings based on data element co-occurrence patterns.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

def calculate_affinity_score(product_group, cooccurrence_matrix, report_count=None):
    """
    Calculate the affinity score for a product group.
    
    The affinity score measures how often elements within a group appear together.
    Higher scores indicate elements that are frequently used together.
    
    Args:
        product_group (list): List of data elements in the product group
        cooccurrence_matrix (pandas.DataFrame): Co-occurrence matrix from co-occurrence analysis
        report_count (int, optional): Total number of reports for normalization.
                                     If None, uses maximum co-occurrence as normalization factor.
    
    Returns:
        float: Affinity score between 0 and 1
    """
    # Ensure all elements in the product group exist in the matrix
    valid_elements = [elem for elem in product_group if elem in cooccurrence_matrix.index]
    
    if len(valid_elements) <= 1:
        return 0.0  # Need at least 2 elements to calculate affinity
    
    # Extract the submatrix for this product group
    submatrix = cooccurrence_matrix.loc[valid_elements, valid_elements]
    
    # Calculate sum of all co-occurrences (excluding self-comparisons)
    total_cooccurrences = 0
    pair_count = 0
    
    for i, elem1 in enumerate(valid_elements):
        for elem2 in valid_elements[i+1:]:  # Start from i+1 to avoid duplicates
            total_cooccurrences += submatrix.loc[elem1, elem2]
            pair_count += 1
    
    if pair_count == 0:
        return 0.0
    
    # Calculate average co-occurrence
    avg_cooccurrence = total_cooccurrences / pair_count
    
    # Normalize by report count if provided, otherwise by max co-occurrence
    if report_count:
        normalization_factor = report_count
    else:
        normalization_factor = cooccurrence_matrix.values.max()
    
    if normalization_factor == 0:
        return 0.0
        
    # Return normalized average (between 0 and 1)
    return avg_cooccurrence / normalization_factor

def identify_common_patterns(df_exploded, min_frequency=2, max_elements=5):
    """
    Identify common usage patterns from report data.
    
    A "pattern" is a set of data elements that frequently appear together in reports.
    
    Args:
        df_exploded (pandas.DataFrame): Exploded dataframe with 'Report' and 'data_element' columns
        min_frequency (int): Minimum number of reports that must use a pattern
        max_elements (int): Maximum number of elements to consider in a pattern
    
    Returns:
        list: List of tuples (pattern_elements, frequency)
    """
    # Group by Report to get all data elements used in each report
    report_groups = df_exploded.groupby("Report")["data_element"].apply(set).reset_index()
    
    # Track pattern frequencies
    pattern_counts = defaultdict(int)
    
    # For each report, generate all possible patterns (element combinations)
    for row in report_groups.itertuples():
        elements = getattr(row, "data_element")
        
        # Generate patterns of different sizes up to max_elements
        for size in range(2, min(max_elements + 1, len(elements) + 1)):
            # Generate all combinations of the specified size
            from itertools import combinations
            for pattern in combinations(sorted(elements), size):
                # Use frozenset as dictionary key since we need an immutable object
                pattern_key = frozenset(pattern)
                pattern_counts[pattern_key] += 1
    
    # Filter patterns by minimum frequency
    common_patterns = [(list(pattern), count) 
                      for pattern, count in pattern_counts.items() 
                      if count >= min_frequency]
    
    # Sort by frequency (descending)
    common_patterns.sort(key=lambda x: x[1], reverse=True)
    
    return common_patterns

def calculate_coverage_ratio(product_groups, common_patterns):
    """
    Calculate the coverage ratio for a set of product groups.
    
    The coverage ratio measures how well the product groups cover common usage patterns.
    A higher ratio means that users can typically find all related elements within a single group.
    
    Args:
        product_groups (list): List of lists, where each inner list contains the data elements for a product group
        common_patterns (list): List of (pattern_elements, frequency) tuples from identify_common_patterns()
    
    Returns:
        dict: Dictionary with overall and detailed coverage metrics
    """
    if not common_patterns:
        return {
            "overall_coverage": 0.0,
            "weighted_coverage": 0.0,
            "pattern_coverage": {}
        }
    
    total_patterns = len(common_patterns)
    covered_patterns = 0
    total_frequency = sum(freq for _, freq in common_patterns)
    weighted_coverage = 0
    
    # Convert product groups to sets for faster lookups
    product_group_sets = [set(group) for group in product_groups]
    
    # Track coverage for each pattern
    pattern_coverage = {}
    
    for pattern, frequency in common_patterns:
        pattern_set = set(pattern)
        
        # Check if the pattern is fully contained in any product group
        is_covered = any(pattern_set.issubset(group) for group in product_group_sets)
        
        if is_covered:
            covered_patterns += 1
            weighted_coverage += frequency
        
        # Store coverage for this pattern
        pattern_key = ", ".join(pattern)
        pattern_coverage[pattern_key] = {
            "covered": is_covered,
            "frequency": frequency,
            "elements": pattern
        }
    
    # Calculate ratios
    overall_coverage = covered_patterns / total_patterns if total_patterns > 0 else 0
    weighted_coverage_ratio = weighted_coverage / total_frequency if total_frequency > 0 else 0
    
    return {
        "overall_coverage": overall_coverage,
        "weighted_coverage": weighted_coverage_ratio,
        "pattern_coverage": pattern_coverage
    }

def calculate_redundancy_score(product_groups):
    """
    Calculate the redundancy score for a set of product groups.
    
    The redundancy score measures unnecessary overlaps between product groups.
    Lower scores indicate less redundancy and are generally better.
    
    Args:
        product_groups (list): List of lists, where each inner list contains the data elements for a product group
    
    Returns:
        dict: Dictionary with redundancy metrics
    """
    # Flatten all product groups to get total unique elements
    all_elements = set()
    for group in product_groups:
        all_elements.update(group)
    
    total_elements = len(all_elements)
    
    if total_elements == 0:
        return {
            "redundancy_score": 0.0,
            "unique_elements": 0,
            "duplicated_elements": 0,
            "element_counts": {}
        }
    
    # Count appearances of each element across groups
    element_counts = defaultdict(int)
    for group in product_groups:
        for element in group:
            element_counts[element] += 1
    
    # Identify elements that appear in multiple groups
    duplicated_elements = {elem for elem, count in element_counts.items() if count > 1}
    num_duplicated = len(duplicated_elements)
    
    # Calculate redundancy score as percentage of duplicated elements
    redundancy_score = num_duplicated / total_elements if total_elements > 0 else 0
    
    return {
        "redundancy_score": redundancy_score,
        "unique_elements": total_elements,
        "duplicated_elements": num_duplicated,
        "element_counts": dict(element_counts)
    }

def evaluate_product_groups(product_groups, cooccurrence_matrix, df_exploded, report_count=None, min_pattern_frequency=2):
    """
    Evaluate a set of product groups using multiple metrics.
    
    This function combines all three metrics (affinity, coverage, redundancy)
    to provide a comprehensive evaluation of potential product groupings.
    
    Args:
        product_groups (list): List of lists, where each inner list contains the data elements for a product group
        cooccurrence_matrix (pandas.DataFrame): Co-occurrence matrix from co-occurrence analysis
        df_exploded (pandas.DataFrame): Exploded dataframe with 'Report' and 'data_element' columns
        report_count (int, optional): Total number of reports (for normalization)
        min_pattern_frequency (int): Minimum frequency for common patterns
    
    Returns:
        dict: Dictionary with all evaluation metrics
    """
    # Identify common usage patterns
    common_patterns = identify_common_patterns(df_exploded, min_frequency=min_pattern_frequency)
    
    # Initialize metrics
    metrics = {
        "group_metrics": [],
        "overall_metrics": {},
    }
    
    # Calculate metrics for each product group
    for i, group in enumerate(product_groups):
        affinity = calculate_affinity_score(group, cooccurrence_matrix, report_count)
        
        metrics["group_metrics"].append({
            "group_id": i,
            "elements": group,
            "size": len(group),
            "affinity_score": affinity
        })
    
    # Calculate overall metrics
    coverage = calculate_coverage_ratio(product_groups, common_patterns)
    redundancy = calculate_redundancy_score(product_groups)
    
    metrics["overall_metrics"] = {
        "coverage_ratio": coverage["overall_coverage"],
        "weighted_coverage": coverage["weighted_coverage"],
        "redundancy_score": redundancy["redundancy_score"],
        "unique_elements": redundancy["unique_elements"],
        "duplicated_elements": redundancy["duplicated_elements"],
        "common_patterns_count": len(common_patterns)
    }
    
    # Add additional details
    metrics["pattern_coverage"] = coverage["pattern_coverage"]
    metrics["element_redundancy"] = redundancy["element_counts"]
    
    # Calculate overall "quality score" (can be customized based on needs)
    # Higher affinity and coverage are better, lower redundancy is better
    avg_affinity = np.mean([g["affinity_score"] for g in metrics["group_metrics"]])
    quality_score = (avg_affinity + coverage["weighted_coverage"]) * (1 - redundancy["redundancy_score"] * 0.5)
    metrics["overall_metrics"]["quality_score"] = quality_score
    
    return metrics