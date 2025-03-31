"""
Validation functions for testing checkpoint operations.

This module provides functions to validate checkpoint data integrity
and test various operations before executing the full program.
"""

import os
import tempfile
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union

import pandas as pd
import numpy as np

from .manager import CheckpointManager
from .param_utils import (
    get_numeric_params, 
    convert_to_numeric_if_possible, 
    is_numeric_convertible, 
    create_parameter_mask
)
from .suggestion_strategies import suggest_parameter_combinations


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


def validate_checkpoint(checkpoint_file: str) -> Tuple[bool, str]:
    """
    Validate that the checkpoint file can be loaded and has required structure.
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Try to create the checkpoint manager
        manager = CheckpointManager(checkpoint_file)
        
        # Check if checkpoint data was loaded
        if not manager.checkpoint_data:
            return False, "Failed to load checkpoint data"
        
        # Check required fields in checkpoint data
        required_fields = ['results', 'pending_combinations', 'param_ranges']
        missing_fields = [field for field in required_fields if field not in manager.checkpoint_data]
        
        if missing_fields:
            return False, f"Checkpoint is missing required fields: {', '.join(missing_fields)}"
        
        # Check that there's at least some results to work with
        results = manager.checkpoint_data.get('results', [])
        if not results:
            return False, "Checkpoint contains no results to analyze"
        
        return True, "Checkpoint validation successful"
        
    except Exception as e:
        return False, f"Error during checkpoint validation: {str(e)}"


def validate_parameter_operations(checkpoint_file: str) -> Tuple[bool, str]:
    """
    Validate parameter operations like type conversion and formatting.
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Get checkpoint data
        manager = CheckpointManager(checkpoint_file)
        param_ranges = manager.get_parameter_ranges()
        results_df = manager.get_results_dataframe()
        
        # Check that we have numeric parameters
        numeric_params = get_numeric_params(results_df)
        if not numeric_params:
            return False, "No numeric parameters found for analysis"
        
        # Test numeric conversion on parameter values
        for param in numeric_params:
            if param in param_ranges:
                param_values = param_ranges[param]
                if isinstance(param_values, list):
                    # Try conversion
                    try:
                        is_numeric = is_numeric_convertible(param_values)
                        converted = convert_to_numeric_if_possible(param_values)
                    except Exception as e:
                        return False, f"Error converting values for parameter {param}: {str(e)}"
        
        # Test parameter mask creation
        try:
            # Get the first row as a test suggestion
            test_suggestion = {p: results_df.iloc[0][p] for p in numeric_params if p in results_df.columns}
            mask = create_parameter_mask(results_df, test_suggestion, numeric_params)
            if not isinstance(mask, pd.Series):
                return False, "Parameter mask creation failed"
        except Exception as e:
            return False, f"Error creating parameter mask: {str(e)}"
            
        return True, "Parameter operations validation successful"
        
    except Exception as e:
        return False, f"Error during parameter operations validation: {str(e)}"


def validate_visualization(checkpoint_file: str) -> Tuple[bool, str]:
    """
    Validate that visualization operations work.
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Get checkpoint data
        manager = CheckpointManager(checkpoint_file)
        results_df = manager.get_results_dataframe()
        
        # Check that required columns exist for visualization
        required_columns = ['quality_score']
        missing_columns = [col for col in required_columns if col not in results_df.columns]
        if missing_columns:
            return False, f"Results are missing required columns for visualization: {', '.join(missing_columns)}"
        
        # Try to create a simple test plot to ensure matplotlib works
        plt.figure(figsize=(2, 2))
        plt.plot([1, 2, 3], [1, 2, 3])
        plt.close()
        
        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Try to save a test plot
            test_file = os.path.join(temp_dir, "test_plot.png")
            plt.figure(figsize=(2, 2))
            plt.plot([1, 2, 3], [1, 2, 3])
            plt.savefig(test_file)
            plt.close()
            
            # Check that the file was created
            if not os.path.exists(test_file):
                return False, "Failed to save test plot file"
            
        return True, "Visualization validation successful"
        
    except Exception as e:
        return False, f"Error during visualization validation: {str(e)}"


def validate_suggestion_generation(checkpoint_file: str) -> Tuple[bool, str]:
    """
    Validate that parameter suggestion generation works.
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Get checkpoint data
        manager = CheckpointManager(checkpoint_file)
        results_df = manager.get_results_dataframe()
        param_ranges = manager.get_parameter_ranges()
        
        # Check that we have numeric parameters
        numeric_params = get_numeric_params(results_df)
        if not numeric_params:
            return False, "No numeric parameters found for suggestion generation"
        
        # Try to generate at least one suggestion
        try:
            suggestions = suggest_parameter_combinations(
                results_df, param_ranges, numeric_params, num_suggestions=1
            )
            
            if not suggestions:
                return False, "Failed to generate any parameter suggestions"
                
        except Exception as e:
            return False, f"Error generating parameter suggestions: {str(e)}"
            
        return True, "Suggestion generation validation successful"
        
    except Exception as e:
        return False, f"Error during suggestion generation validation: {str(e)}"


def validate_output_directory(output_dir: str) -> Tuple[bool, str]:
    """
    Validate that the output directory can be created and is writable.
    
    Args:
        output_dir: Output directory to validate
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if the directory exists
        if not os.path.isdir(output_dir):
            return False, f"Failed to create output directory: {output_dir}"
        
        # Try to create a test file
        test_file = os.path.join(output_dir, "test_write.txt")
        try:
            with open(test_file, 'w') as f:
                f.write("Test write")
            
            # Clean up the test file
            os.remove(test_file)
            
        except Exception as e:
            return False, f"Directory is not writable: {str(e)}"
            
        return True, "Output directory validation successful"
        
    except Exception as e:
        return False, f"Error during output directory validation: {str(e)}"


def run_all_validations(checkpoint_file: str, output_dir: str) -> Dict[str, Dict[str, Union[bool, str]]]:
    """
    Run all validation checks and return the results.
    
    Args:
        checkpoint_file: Path to checkpoint file
        output_dir: Output directory
        
    Returns:
        Dictionary with validation results
    """
    validations = {
        "checkpoint": validate_checkpoint(checkpoint_file),
        "parameter_operations": validate_parameter_operations(checkpoint_file),
        "visualization": validate_visualization(checkpoint_file),
        "suggestion_generation": validate_suggestion_generation(checkpoint_file),
        "output_directory": validate_output_directory(output_dir)
    }
    
    # Format the results
    results = {}
    for name, (success, message) in validations.items():
        results[name] = {
            "success": success,
            "message": message
        }
    
    return results


def validate_and_print(checkpoint_file: str, output_dir: str) -> bool:
    """
    Run all validations and print the results.
    
    Args:
        checkpoint_file: Path to checkpoint file
        output_dir: Output directory
        
    Returns:
        True if all validations passed, False otherwise
    """
    print("\n=== Running Pre-execution Validation ===\n")
    
    validation_results = run_all_validations(checkpoint_file, output_dir)
    
    all_passed = True
    for name, result in validation_results.items():
        success = result["success"]
        message = result["message"]
        
        status = "PASSED" if success else "FAILED"
        print(f"{name.replace('_', ' ').title()} Validation: {status}")
        print(f"  {message}")
        
        if not success:
            all_passed = False
    
    print("\n=== Validation Summary ===")
    if all_passed:
        print("All validations passed! The program should run without issues.")
    else:
        print("Some validations failed. Running the program may result in errors.")
    
    return all_passed