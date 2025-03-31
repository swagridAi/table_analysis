# optimization_validation.py
"""
Validation functions for parameter optimization process.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

def validate_parameter_ranges(param_ranges: Dict[str, List[Any]]) -> Tuple[bool, str]:
    """Validate parameter ranges for optimization."""
    if not param_ranges:
        return False, "Parameter ranges dictionary is empty"
    
    for param_name, values in param_ranges.items():
        if not isinstance(values, list) or not values:
            return False, f"Parameter {param_name} has invalid range: must be a non-empty list"
            
        # Check for type consistency within each parameter range
        try:
            if all(isinstance(v, (int, float)) for v in values):
                # All numeric, check for correct ordering
                if min(values) == max(values):
                    return False, f"Parameter {param_name} has only one value: {values[0]}"
            elif all(isinstance(v, str) for v in values):
                # All strings, check if they're valid
                if len(values) < 2:
                    return False, f"Parameter {param_name} needs multiple values for optimization"
        except Exception as e:
            return False, f"Error validating parameter {param_name}: {str(e)}"
            
    return True, "Parameter ranges validation successful"

def validate_input_file(input_file: str) -> Tuple[bool, str]:
    """Validate that the input file exists and is readable."""
    if not os.path.exists(input_file):
        return False, f"Input file not found: {input_file}"
        
    try:
        # Try to open and read a small portion of the file
        with open(input_file, 'r') as f:
            f.read(1024)
        return True, "Input file validation successful"
    except Exception as e:
        return False, f"Error reading input file: {str(e)}"

def validate_output_directory(output_dir: str) -> Tuple[bool, str]:
    """Validate that the output directory can be created and is writable."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.isdir(output_dir):
            return False, f"Failed to create output directory: {output_dir}"
        
        # Test write access
        test_file = os.path.join(output_dir, "test_write.txt")
        try:
            with open(test_file, 'w') as f:
                f.write("Test write")
            os.remove(test_file)
        except Exception as e:
            return False, f"Directory is not writable: {str(e)}"
            
        return True, "Output directory validation successful"
    except Exception as e:
        return False, f"Error during output directory validation: {str(e)}"

def validate_checkpoint_if_exists(output_dir: str) -> Tuple[bool, str]:
    """Validate the checkpoint file if it exists for resuming."""
    checkpoint_file = os.path.join(output_dir, "search_checkpoint.pkl")
    
    if not os.path.exists(checkpoint_file):
        return True, "No existing checkpoint found. Will start new optimization."
        
    try:
        import pickle
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            
        # Check expected fields
        required_fields = ['results', 'pending_combinations', 'param_ranges']
        missing_fields = [f for f in required_fields if f not in checkpoint_data]
        
        if missing_fields:
            return False, f"Existing checkpoint is missing required fields: {', '.join(missing_fields)}"
            
        return True, "Existing checkpoint validated for resuming"
    except Exception as e:
        return False, f"Error validating existing checkpoint: {str(e)}"
        
def run_optimization_validation(
    input_file: str, 
    output_dir: str,
    param_ranges: Dict[str, List[Any]]
) -> bool:
    """Run all validations and print results."""
    print("\n=== Running Pre-optimization Validation ===\n")
    
    validations = [
        ("Input File", validate_input_file(input_file)),
        ("Output Directory", validate_output_directory(output_dir)),
        ("Parameter Ranges", validate_parameter_ranges(param_ranges)),
        ("Checkpoint", validate_checkpoint_if_exists(output_dir))
    ]
    
    all_passed = True
    for name, (success, message) in validations:
        status = "PASSED" if success else "FAILED"
        print(f"{name} Validation: {status}")
        print(f"  {message}")
        
        if not success:
            all_passed = False
    
    print("\n=== Validation Summary ===")
    if all_passed:
        print("All validations passed! Optimization can proceed.")
    else:
        print("Some validations failed. Optimization may encounter errors.")
    
    return all_passed