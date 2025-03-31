"""
Core checkpoint manager functionality.

This module provides the primary CheckpointManager class for loading, analyzing,
and modifying parameter optimization checkpoints.
"""

import os
import pickle
import json
from typing import Dict, List, Optional, Any, Union

import pandas as pd

from .param_utils import get_numeric_params, create_parameter_mask


class CheckpointManager:
    """
    Core manager for optimization checkpoints.
    
    This class handles loading checkpoint files, retrieving optimization results,
    and modifying checkpoint data.
    """
    
    def __init__(self, checkpoint_file: str):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_file: Path to the checkpoint file
        """
        self.checkpoint_file = checkpoint_file
        self.checkpoint_data = None
        
        # Load the checkpoint
        self._load_checkpoint()
    
    def _load_checkpoint(self) -> bool:
        """
        Load the checkpoint data from file.
        
        Returns:
            True if the checkpoint was loaded successfully, False otherwise
        """
        if not os.path.exists(self.checkpoint_file):
            print(f"Checkpoint file not found: {self.checkpoint_file}")
            return False
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                self.checkpoint_data = pickle.load(f)
            
            print(f"Checkpoint loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the optimization process.
        
        Returns:
            Dictionary with status information including completion percentage,
            number of evaluated combinations, and current optimal parameters
        """
        if not self.checkpoint_data:
            return {"status": "No checkpoint data loaded"}
        
        results = self.checkpoint_data.get('results', [])
        pending = self.checkpoint_data.get('pending_combinations', [])
        optimal = self.checkpoint_data.get('optimal_params', None)
        iterations = self.checkpoint_data.get('iterations_completed', 0)
        timestamp = self.checkpoint_data.get('timestamp', None)
        
        total_combinations = len(results) + len(pending)
        completion_percentage = 0
        if total_combinations > 0:
            completion_percentage = round(len(results) / total_combinations * 100, 2)
        
        status = {
            "timestamp": timestamp,
            "iterations_completed": iterations,
            "total_combinations": total_combinations,
            "completed_combinations": len(results),
            "pending_combinations": len(pending),
            "completion_percentage": completion_percentage,
            "has_optimal_params": optimal is not None,
        }
        
        if optimal:
            status["current_optimal"] = {
                k: v for k, v in optimal.items() 
                if k not in ['combination_id'] and not k.startswith('_')
            }
        
        return status
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get a DataFrame with the results of evaluated combinations.
        
        Returns:
            DataFrame with optimization results
        """
        if not self.checkpoint_data or not self.checkpoint_data.get('results'):
            print("No results found in checkpoint")
            return pd.DataFrame()
        
        return pd.DataFrame(self.checkpoint_data['results'])
    
    def get_pending_dataframe(self) -> pd.DataFrame:
        """
        Get a DataFrame with the pending parameter combinations.
        
        Returns:
            DataFrame with pending combinations
        """
        if not self.checkpoint_data or not self.checkpoint_data.get('pending_combinations'):
            print("No pending combinations found in checkpoint")
            return pd.DataFrame()
        
        return pd.DataFrame(self.checkpoint_data['pending_combinations'])
    
    def get_optimal_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Get the current optimal parameters.
        
        Returns:
            Dictionary with optimal parameters, or None if not available
        """
        if not self.checkpoint_data:
            return None
        
        return self.checkpoint_data.get('optimal_params')
    
    def get_parameter_ranges(self) -> Dict[str, List[Any]]:
        """
        Get the parameter ranges from the checkpoint.
        
        Returns:
            Dictionary with parameter ranges
        """
        if not self.checkpoint_data:
            return {}
        
        return self.checkpoint_data.get('param_ranges', {})
    
    def export_results(self, output_dir: str) -> bool:
        """
        Export the current results to CSV and the optimal parameters to JSON.
        
        Args:
            output_dir: Directory to save the exports
        
        Returns:
            True if exports were created, False otherwise
        """
        if not self.checkpoint_data:
            print("No checkpoint data loaded")
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export results to CSV
        results_df = self.get_results_dataframe()
        if not results_df.empty:
            results_file = os.path.join(output_dir, "optimization_results.csv")
            results_df.to_csv(results_file, index=False)
            print(f"Results exported to {results_file}")
        
        # Export optimal parameters to JSON
        optimal_params = self.get_optimal_parameters()
        if optimal_params:
            optimal_file = os.path.join(output_dir, "optimal_parameters.json")
            with open(optimal_file, 'w') as f:
                json.dump(optimal_params, f, indent=2)
            print(f"Optimal parameters exported to {optimal_file}")
        
        # Export status to JSON
        status = self.get_status()
        status_file = os.path.join(output_dir, "search_status.json")
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        print(f"Search status exported to {status_file}")
        
        return True
    
    def modify_checkpoint(
        self, 
        add_combinations: Optional[pd.DataFrame] = None, 
        remove_combinations: Optional[pd.DataFrame] = None
    ) -> bool:
        """
        Modify the checkpoint by adding or removing parameter combinations.
        
        Args:
            add_combinations: DataFrame with combinations to add to pending list
            remove_combinations: DataFrame with combinations to remove from pending list
            
        Returns:
            True if the checkpoint was modified, False otherwise
        """
        if not self.checkpoint_data:
            print("No checkpoint data loaded")
            return False
        
        modified = False
        
        # Add combinations to pending list
        if add_combinations is not None and not add_combinations.empty:
            # Convert DataFrame to list of dictionaries
            combinations_to_add = add_combinations.to_dict('records')
            
            # Get current pending combinations
            pending = self.checkpoint_data.get('pending_combinations', [])
            
            # Add new combinations
            pending.extend(combinations_to_add)
            
            # Update checkpoint data
            self.checkpoint_data['pending_combinations'] = pending
            modified = True
            print(f"Added {len(combinations_to_add)} combinations to pending list")
        
        # Remove combinations from pending list
        if remove_combinations is not None and not remove_combinations.empty:
            modified = self._remove_combinations_from_pending(remove_combinations)
        
        # Save modified checkpoint if changes were made
        if modified:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(self.checkpoint_data, f)
            print(f"Modified checkpoint saved to {self.checkpoint_file}")
        
        return modified
    
    def _remove_combinations_from_pending(
        self, 
        remove_combinations: pd.DataFrame
    ) -> bool:
        """
        Remove specified combinations from the pending list.
        
        Args:
            remove_combinations: DataFrame with combinations to remove
            
        Returns:
            True if combinations were removed, False otherwise
        """
        # Get current pending combinations
        pending = self.checkpoint_data.get('pending_combinations', [])
        if not pending:
            print("No pending combinations to remove")
            return False
        
        # Convert pending to DataFrame for easier comparison
        pending_df = pd.DataFrame(pending)
        
        # Identify common parameters for comparison
        numeric_params = [
            'community_resolution',
            'min_pattern_frequency',
            'quality_weight_coverage',
            'quality_weight_redundancy'
        ]
        
        # Filter to parameters that exist in both DataFrames
        common_params = [p for p in numeric_params 
                        if p in pending_df.columns and p in remove_combinations.columns]
        
        if not common_params:
            print("No common parameters found for comparison")
            return False
        
        # Find indices of combinations to remove
        indices_to_remove = []
        
        for i, row in enumerate(pending):
            for _, remove_row in remove_combinations.iterrows():
                if all(row.get(param) == remove_row[param] 
                      for param in common_params if param in row):
                    indices_to_remove.append(i)
                    break
        
        # Remove combinations in reverse order to avoid index issues
        for i in sorted(indices_to_remove, reverse=True):
            del pending[i]
        
        # Update checkpoint data
        self.checkpoint_data['pending_combinations'] = pending
        
        print(f"Removed {len(indices_to_remove)} combinations from pending list")
        return len(indices_to_remove) > 0