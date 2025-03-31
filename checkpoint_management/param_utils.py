"""
Parameter utility functions for handling different parameter types.

This module provides helper functions for working with different parameter types,
performing type conversions, and validating parameter values.
"""

from typing import Dict, List, Optional, Any, Union

import pandas as pd


def is_numeric_convertible(values: List[Any]) -> bool:
    """
    Check if all values in a list are numeric or can be converted to numeric.
    
    Args:
        values: List of values to check
        
    Returns:
        True if all values are numeric or can be converted to numeric
    """
    if not values:
        return False
        
    for v in values:
        if not (isinstance(v, (int, float)) or 
               (isinstance(v, str) and v.replace('.', '', 1).isdigit())):
            return False
            
    return True


def convert_to_numeric_if_possible(values: List[Any]) -> List[Union[float, Any]]:
    """
    Convert a list of values to numeric if possible.
    
    Args:
        values: List of values to convert
        
    Returns:
        List with numeric values where possible, original values otherwise
    """
    result = []
    
    for v in values:
        if isinstance(v, (int, float)):
            result.append(float(v))
        elif isinstance(v, str) and v.replace('.', '', 1).isdigit():
            result.append(float(v))
        else:
            result.append(v)
            
    return result


def get_numeric_params(df: pd.DataFrame) -> List[str]:
    """
    Get the list of numeric parameters that exist in the dataframe.
    
    Args:
        df: DataFrame with parameters
        
    Returns:
        List of numeric parameter names
    """
    # Standard numeric parameters to check
    candidate_params = [
        'community_resolution',
        'min_pattern_frequency',
        'quality_weight_coverage', 
        'quality_weight_redundancy'
    ]
    
    # Filter to parameters that exist and vary in the results
    return [p for p in candidate_params if p in df.columns and df[p].nunique() > 1]


def create_parameter_mask(
    results_df: pd.DataFrame, 
    suggestion: Dict[str, Any], 
    params: List[str]
) -> pd.Series:
    """
    Create a boolean mask for matching a parameter combination in results.
    
    Args:
        results_df: DataFrame with results
        suggestion: Dictionary with parameter values to match
        params: List of parameters to check
        
    Returns:
        Boolean mask series indicating which rows match all parameters
    """
    # Start with all True
    mask = pd.Series(True, index=results_df.index)
    
    for param in params:
        if param in suggestion and param in results_df.columns:
            # Handle different types for comparison
            if isinstance(suggestion[param], (int, float)) and results_df[param].dtype == 'object':
                # Try to convert strings to numbers for comparison
                try:
                    param_values_converted = results_df[param].apply(
                        lambda x: float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x
                    )
                    mask = mask & (param_values_converted == suggestion[param])
                except:
                    # If conversion fails, compare as is
                    mask = mask & (results_df[param] == suggestion[param])
            else:
                mask = mask & (results_df[param] == suggestion[param])
                
    return mask


def format_parameter_value(value: Any, reference_value: Any) -> Any:
    """
    Format a parameter value to match the type of a reference value.
    
    Args:
        value: Value to format
        reference_value: Reference value to match type
        
    Returns:
        Formatted value
    """
    # If reference is a string that represents a number and value is numeric
    if isinstance(reference_value, str) and reference_value.replace('.', '', 1).isdigit() and isinstance(value, (int, float)):
        # Format as integer or float string based on reference
        if '.' in reference_value:
            return str(value)
        else:
            return str(int(value))
    return value


def calculate_parameter_correlations(
    results_df: pd.DataFrame, 
    numeric_params: List[str]
) -> Dict[str, float]:
    """
    Calculate correlations between parameters and quality score.
    
    Args:
        results_df: DataFrame with results
        numeric_params: List of numeric parameters
        
    Returns:
        Dictionary mapping parameter names to correlation values
    """
    correlation_data = {}
    
    for param in numeric_params:
        try:
            # Convert parameter to numeric if it's stored as string but represents a number
            if results_df[param].dtype == 'object':
                param_values = pd.to_numeric(results_df[param], errors='coerce')
                if not param_values.isna().all():  # If conversion was successful
                    correlation = param_values.corr(results_df['quality_score'])
                else:
                    correlation = 0  # No correlation for non-numeric data
            else:
                correlation = results_df[param].corr(results_df['quality_score'])
            
            correlation_data[param] = correlation
        except Exception:
            correlation_data[param] = 0
            
    return correlation_data