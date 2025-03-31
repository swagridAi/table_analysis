"""
Strategies for suggesting new parameter combinations to explore.

This module provides algorithms for generating promising parameter combinations
to explore next during optimization.
"""

from typing import Dict, List, Optional, Any, Union, Tuple

import pandas as pd
import numpy as np

from .param_utils import (
    is_numeric_convertible, 
    convert_to_numeric_if_possible, 
    create_parameter_mask, 
    format_parameter_value,
    calculate_parameter_correlations
)


def suggest_parameter_combinations(
    results_df: pd.DataFrame,
    param_ranges: Dict[str, List[Any]],
    numeric_params: List[str],
    num_suggestions: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate suggested parameter combinations using multiple strategies.
    
    Args:
        results_df: DataFrame with results
        param_ranges: Dictionary with parameter ranges
        numeric_params: List of numeric parameters
        num_suggestions: Number of suggestions to generate
        
    Returns:
        List of suggested parameter combinations
    """
    # Get best point for reference
    best_idx = results_df['quality_score'].idxmax()
    best_point = results_df.loc[best_idx]
    
    # Create suggestions using different strategies
    suggestions = []
    suggestions.extend(strategy_explore_around_best(
        results_df, best_point, numeric_params, param_ranges
    ))
    suggestions.extend(strategy_follow_correlation_trends(
        results_df, numeric_params, param_ranges
    ))
    suggestions.extend(strategy_fill_parameter_gaps(
        results_df, best_point, numeric_params, param_ranges
    ))
    
    # Limit to requested number of suggestions
    return suggestions[:num_suggestions]


def strategy_explore_around_best(
    results_df: pd.DataFrame, 
    best_point: pd.Series, 
    numeric_params: List[str], 
    param_ranges: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """
    Strategy 1: Explore around the current best point.
    
    This strategy suggests parameter combinations that are close to the current best
    parameters, exploring the neighborhood of the optimal point.
    
    Args:
        results_df: DataFrame with results
        best_point: Best parameter combination found so far
        numeric_params: List of numeric parameters
        param_ranges: Dictionary with parameter ranges
        
    Returns:
        List of suggested parameter combinations
    """
    suggestions = []
    
    for param in numeric_params:
        current_value = best_point[param]
        param_values = param_ranges[param]
        
        if not isinstance(param_values, list):
            continue
            
        try:
            # Handle numeric conversion
            if is_numeric_convertible(param_values):
                # Convert to numeric for consistent comparisons
                numeric_values = convert_to_numeric_if_possible(param_values)
                sorted_values = sorted(numeric_values)
                
                # Convert current value to numeric if it's a string representation of a number
                if isinstance(current_value, str) and current_value.replace('.', '', 1).isdigit():
                    current_value_numeric = float(current_value)
                else:
                    current_value_numeric = float(current_value)
            else:
                # Mixed types, convert all to strings
                sorted_values = sorted([str(v) for v in param_values])
                current_value_numeric = str(current_value)
            
            # Find index of current value
            current_idx = sorted_values.index(current_value_numeric)
            
            # Try values on either side if they exist
            for offset in [-2, -1, 1, 2]:
                new_idx = current_idx + offset
                if 0 <= new_idx < len(sorted_values):
                    new_value = sorted_values[new_idx]
                    
                    # Convert back to original type if needed
                    new_value = format_parameter_value(new_value, param_values[0])
                    
                    # Create a new suggestion
                    suggestion = create_suggestion_from_point(
                        best_point, param, new_value, "Explore around best point", 
                        results_df, param_ranges, numeric_params
                    )
                    
                    if suggestion:
                        suggestions.append(suggestion)
        except (ValueError, TypeError):
            # Skip this parameter if there's an error
            continue
            
    return suggestions


def strategy_follow_correlation_trends(
    results_df: pd.DataFrame, 
    numeric_params: List[str], 
    param_ranges: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """
    Strategy 2: Follow correlation trends in high-performing regions.
    
    This strategy analyzes how parameters correlate with quality score and
    suggests moving in the direction of positive correlation.
    
    Args:
        results_df: DataFrame with results
        numeric_params: List of numeric parameters
        param_ranges: Dictionary with parameter ranges
        
    Returns:
        List of suggested parameter combinations
    """
    suggestions = []
    
    # Get top performing combinations
    top_n = min(5, len(results_df))
    top_performers = results_df.nlargest(top_n, 'quality_score')
    
    # Calculate parameter correlations with quality score
    correlation_data = calculate_parameter_correlations(results_df, numeric_params)
    
    # For each top performer, suggest moving in the direction of correlation
    for _, performer in top_performers.iterrows():
        for param in numeric_params:
            corr_value = correlation_data.get(param, 0)
            current_value = performer[param]
            param_values = param_ranges[param]
            
            if not isinstance(param_values, list):
                continue
                
            try:
                # Handle numeric conversion
                if is_numeric_convertible(param_values):
                    # Convert to numeric for consistent comparisons
                    numeric_values = convert_to_numeric_if_possible(param_values)
                    sorted_values = sorted(numeric_values)
                    
                    # Convert current value to numeric if it's a string representation of a number
                    if isinstance(current_value, str) and current_value.replace('.', '', 1).isdigit():
                        current_value_numeric = float(current_value)
                    else:
                        current_value_numeric = float(current_value)
                else:
                    # Mixed types, convert all to strings
                    sorted_values = sorted([str(v) for v in param_values])
                    current_value_numeric = str(current_value)
                
                # Find index of current value
                current_idx = sorted_values.index(current_value_numeric)
                
                # Determine direction based on correlation
                if corr_value > 0 and current_idx < len(sorted_values) - 1:
                    new_value = sorted_values[current_idx + 1]
                elif corr_value < 0 and current_idx > 0:
                    new_value = sorted_values[current_idx - 1]
                else:
                    continue
                
                # Convert back to original type if needed
                new_value = format_parameter_value(new_value, param_values[0])
                
                # Create a new suggestion
                suggestion = create_suggestion_from_point(
                    performer, param, new_value, "Follow correlation trend", 
                    results_df, param_ranges, numeric_params
                )
                
                if suggestion:
                    suggestions.append(suggestion)
            except (ValueError, TypeError):
                # Skip this parameter if there's an error
                continue
    
    return suggestions


def strategy_fill_parameter_gaps(
    results_df: pd.DataFrame, 
    best_point: pd.Series, 
    numeric_params: List[str], 
    param_ranges: Dict[str, List[Any]]
) -> List[Dict[str, Any]]:
    """
    Strategy 3: Fill gaps in parameter space.
    
    This strategy identifies parameter values that haven't been thoroughly explored
    and suggests combinations to fill these gaps.
    
    Args:
        results_df: DataFrame with results
        best_point: Best parameter combination found so far
        numeric_params: List of numeric parameters
        param_ranges: Dictionary with parameter ranges
        
    Returns:
        List of suggested parameter combinations
    """
    suggestions = []
    
    for param in numeric_params:
        param_values = param_ranges[param]
        
        if not isinstance(param_values, list):
            continue
            
        try:
            # Handle numeric conversion for value counting
            if is_numeric_convertible(param_values):
                # Convert to numeric for consistent comparisons
                param_values_numeric = convert_to_numeric_if_possible(param_values)
                
                # Count occurrences of each value
                try:
                    value_counts = results_df[param].astype(float).value_counts()
                except:
                    # Use values as is if conversion fails
                    value_counts = results_df[param].value_counts()
            else:
                # Use values as is
                value_counts = results_df[param].value_counts()
                param_values_numeric = param_values
            
            # Find underexplored values
            for value in param_values_numeric:
                if value not in value_counts or value_counts[value] < value_counts.mean() / 2:
                    # Use median values for other parameters
                    suggestion = create_gap_filling_suggestion(
                        results_df, best_point, param, value, param_ranges, numeric_params
                    )
                    
                    if suggestion:
                        suggestions.append(suggestion)
        except (ValueError, TypeError):
            # Skip this parameter if there's an error
            continue
    
    return suggestions


def create_suggestion_from_point(
    base_point: pd.Series, 
    param_to_change: str, 
    new_value: Any, 
    strategy: str,
    results_df: pd.DataFrame,
    param_ranges: Dict[str, List[Any]],
    numeric_params: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Create a suggestion by modifying one parameter of a base point.
    
    Args:
        base_point: Base parameter combination
        param_to_change: Parameter to modify
        new_value: New value for the parameter
        strategy: Description of the suggestion strategy
        results_df: DataFrame with results
        param_ranges: Dictionary with parameter ranges
        numeric_params: List of numeric parameters
        
    Returns:
        Suggestion dictionary or None if the combination already exists
    """
    # Copy relevant parameters from base point
    suggestion = {p: base_point[p] for p in results_df.columns 
                 if p in param_ranges and p != 'combination_id'}
    
    # Change the specified parameter
    suggestion[param_to_change] = new_value
    suggestion['strategy'] = f"{strategy} ({param_to_change})"
    
    # Check if this combination already exists
    mask = create_parameter_mask(results_df, suggestion, numeric_params)
    
    # Only return if this is a new combination
    if not mask.any():
        return suggestion
    return None


def create_gap_filling_suggestion(
    results_df: pd.DataFrame, 
    best_point: pd.Series, 
    param: str, 
    value: Any, 
    param_ranges: Dict[str, List[Any]], 
    numeric_params: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Create a suggestion to fill a gap in parameter space.
    
    Args:
        results_df: DataFrame with results
        best_point: Best parameter combination found so far
        param: Parameter to modify
        value: Underexplored value to use
        param_ranges: Dictionary with parameter ranges
        numeric_params: List of numeric parameters
        
    Returns:
        Suggestion dictionary or None if creation fails
    """
    suggestion = {}
    
    # Set the value for the target parameter
    suggestion[param] = value
    
    # Use median values from successful combinations for other parameters
    for p in numeric_params:
        if p == param:
            continue
            
        # Use median of successful combinations for this parameter
        top_half = results_df[results_df['quality_score'] > results_df['quality_score'].median()]
        
        if not top_half.empty:
            # Convert to numeric if possible
            try:
                if top_half[p].dtype == 'object':
                    numeric_values = pd.to_numeric(top_half[p], errors='coerce')
                    if not numeric_values.isna().all():
                        suggestion[p] = numeric_values.median()
                    else:
                        suggestion[p] = top_half[p].mode()[0]
                else:
                    suggestion[p] = top_half[p].median()
            except:
                suggestion[p] = top_half[p].mode()[0]
        else:
            try:
                if results_df[p].dtype == 'object':
                    numeric_values = pd.to_numeric(results_df[p], errors='coerce')
                    if not numeric_values.isna().all():
                        suggestion[p] = numeric_values.median()
                    else:
                        suggestion[p] = results_df[p].mode()[0]
                else:
                    suggestion[p] = results_df[p].median()
            except:
                suggestion[p] = results_df[p].mode()[0]
    
    # Add categorical parameters from the best combination
    for p in param_ranges:
        if p not in numeric_params and p in best_point:
            suggestion[p] = best_point[p]
    
    suggestion['strategy'] = f"Fill gap in parameter space ({param}={value})"
    
    # Check if this combination already exists
    mask = create_parameter_mask(results_df, suggestion, numeric_params)
    
    # Only return if this is a new combination
    if not mask.any():
        return suggestion
    return None