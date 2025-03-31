#!/usr/bin/env python3
"""
Checkpoint manager for parameter optimization.

This module provides utilities for managing and analyzing optimization checkpoints,
allowing users to examine search progress, visualize parameter spaces, and identify
promising regions for more targeted exploration.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import argparse
from datetime import datetime

class CheckpointManager:
    """
    Utility for managing and analyzing parameter optimization checkpoints.
    """
    
    def __init__(self, checkpoint_file):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_file (str): Path to the checkpoint file
        """
        self.checkpoint_file = checkpoint_file
        self.checkpoint_data = None
        
        # Load the checkpoint
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """
        Load the checkpoint data.
        
        Returns:
            bool: True if the checkpoint was loaded successfully, False otherwise
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
    
    def get_status(self):
        """
        Get the status of the optimization process.
        
        Returns:
            dict: Dictionary with status information
        """
        if not self.checkpoint_data:
            return {"status": "No checkpoint data loaded"}
        
        results = self.checkpoint_data.get('results', [])
        pending = self.checkpoint_data.get('pending_combinations', [])
        optimal = self.checkpoint_data.get('optimal_params', None)
        iterations = self.checkpoint_data.get('iterations_completed', 0)
        timestamp = self.checkpoint_data.get('timestamp', None)
        
        status = {
            "timestamp": timestamp,
            "iterations_completed": iterations,
            "total_combinations": len(results) + len(pending),
            "completed_combinations": len(results),
            "pending_combinations": len(pending),
            "completion_percentage": round(len(results) / (len(results) + len(pending)) * 100, 2) if (len(results) + len(pending)) > 0 else 0,
            "has_optimal_params": optimal is not None,
        }
        
        if optimal:
            status["current_optimal"] = {
                k: v for k, v in optimal.items() 
                if k not in ['combination_id'] and not k.startswith('_')
            }
        
        return status
    
    def get_results_dataframe(self):
        """
        Get a DataFrame with the results of evaluated combinations.
        
        Returns:
            pd.DataFrame: DataFrame with results
        """
        if not self.checkpoint_data or not self.checkpoint_data.get('results'):
            print("No results found in checkpoint")
            return pd.DataFrame()
        
        return pd.DataFrame(self.checkpoint_data['results'])
    
    def get_pending_dataframe(self):
        """
        Get a DataFrame with the pending parameter combinations.
        
        Returns:
            pd.DataFrame: DataFrame with pending combinations
        """
        if not self.checkpoint_data or not self.checkpoint_data.get('pending_combinations'):
            print("No pending combinations found in checkpoint")
            return pd.DataFrame()
        
        return pd.DataFrame(self.checkpoint_data['pending_combinations'])
    
    def get_optimal_parameters(self):
        """
        Get the current optimal parameters.
        
        Returns:
            dict: Dictionary with optimal parameters, or None if not available
        """
        if not self.checkpoint_data:
            return None
        
        return self.checkpoint_data.get('optimal_params')
    
    def visualize_parameter_space_coverage(self, output_dir=None):
        """
        Visualize which regions of the parameter space have been explored.
        
        Args:
            output_dir (str, optional): Directory to save visualizations. If None, just displays plots.
        
        Returns:
            bool: True if visualizations were created, False otherwise
        """
        if not self.checkpoint_data:
            print("No checkpoint data loaded")
            return False
        
        # Get results and parameter ranges
        results_df = self.get_results_dataframe()
        
        if results_df.empty:
            print("No results available to visualize")
            return False
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Determine which parameters to visualize
        numeric_params = [
            'community_resolution', 
            'min_pattern_frequency',
            'quality_weight_coverage', 
            'quality_weight_redundancy'
        ]
        
        # Filter to parameters that exist and vary in the results
        numeric_params = [p for p in numeric_params if p in results_df.columns and results_df[p].nunique() > 1]
        
        if len(numeric_params) < 2:
            print("Not enough numeric parameters to visualize parameter space")
            return False
        
        # Create scatterplots for pairs of parameters
        for i, param1 in enumerate(numeric_params):
            for param2 in numeric_params[i+1:]:
                plt.figure(figsize=(10, 8))
                
                # Create scatterplot colored by quality score
                scatter = plt.scatter(
                    results_df[param1], 
                    results_df[param2], 
                    c=results_df['quality_score'],
                    cmap='viridis',
                    alpha=0.8,
                    s=100,
                    edgecolors='k',
                    linewidths=0.5
                )
                
                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label('Quality Score')
                
                # Mark optimal point
                best_idx = results_df['quality_score'].idxmax()
                best_x = results_df.loc[best_idx, param1]
                best_y = results_df.loc[best_idx, param2]
                plt.scatter(
                    best_x, best_y,
                    s=200,
                    marker='*',
                    color='red',
                    edgecolors='k',
                    linewidths=1.5,
                    label='Optimal'
                )
                
                # Add parameter space boundaries if we know them
                if 'param_ranges' in self.checkpoint_data:
                    param_ranges = self.checkpoint_data['param_ranges']
                    if param1 in param_ranges and param2 in param_ranges:
                        x_range = param_ranges[param1]
                        y_range = param_ranges[param2]
                        
                        if isinstance(x_range, list) and isinstance(y_range, list):
                            # Convert to numeric if possible
                            try:
                                x_min = min(float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x for x in x_range)
                                x_max = max(float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x for x in x_range)
                                y_min = min(float(y) if isinstance(y, str) and y.replace('.', '', 1).isdigit() else y for y in y_range)
                                y_max = max(float(y) if isinstance(y, str) and y.replace('.', '', 1).isdigit() else y for y in y_range)
                                
                                # Add slightly transparent background to show full parameter space
                                plt.fill_betweenx([y_min, y_max], x_min, x_max, alpha=0.1, color='gray')
                                
                                # Set axis limits to parameter space
                                plt.xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
                                plt.ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
                            except (TypeError, ValueError):
                                # If conversion fails, skip parameter space boundaries
                                pass
                
                plt.title(f'Parameter Space Coverage: {param1} vs {param2}')
                plt.xlabel(param1)
                plt.ylabel(param2)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f'parameter_space_{param1}_vs_{param2}.png'), dpi=300)
                else:
                    plt.show()
                
                plt.close()
        
        # Create a 3D visualization if we have at least 3 numeric parameters
        if len(numeric_params) >= 3:
            from mpl_toolkits.mplot3d import Axes3D
            
            param1, param2, param3 = numeric_params[:3]
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create 3D scatter plot
            scatter = ax.scatter(
                results_df[param1],
                results_df[param2],
                results_df[param3],
                c=results_df['quality_score'],
                cmap='viridis',
                s=50,
                alpha=0.8
            )
            
            # Mark optimal point
            best_idx = results_df['quality_score'].idxmax()
            best_x = results_df.loc[best_idx, param1]
            best_y = results_df.loc[best_idx, param2]
            best_z = results_df.loc[best_idx, param3]
            
            ax.scatter(
                best_x, best_y, best_z,
                s=200,
                marker='*',
                color='red',
                label='Optimal'
            )
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Quality Score')
            
            ax.set_title(f'3D Parameter Space Coverage')
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_zlabel(param3)
            ax.legend()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, '3d_parameter_space.png'), dpi=300)
            else:
                plt.show()
            
            plt.close()
        
        # Create a visualization of remaining parameter space
        # This will only work if we have both results and pending combinations
        pending_df = self.get_pending_dataframe()
        
        if not pending_df.empty and len(numeric_params) >= 2:
            param1, param2 = numeric_params[:2]
            
            # Check if these parameters exist in the pending dataframe
            if param1 in pending_df.columns and param2 in pending_df.columns:
                plt.figure(figsize=(10, 8))
                
                # Plot explored points
                plt.scatter(
                    results_df[param1],
                    results_df[param2],
                    c='blue',
                    alpha=0.7,
                    s=80,
                    label='Explored'
                )
                
                # Plot pending points
                plt.scatter(
                    pending_df[param1],
                    pending_df[param2],
                    c='gray',
                    alpha=0.3,
                    s=30,
                    label='Pending'
                )
                
                # Mark optimal point
                best_idx = results_df['quality_score'].idxmax()
                best_x = results_df.loc[best_idx, param1]
                best_y = results_df.loc[best_idx, param2]
                
                plt.scatter(
                    best_x, best_y,
                    s=200,
                    marker='*',
                    color='red',
                    label='Current Optimal'
                )
                
                plt.title('Explored vs Pending Parameter Combinations')
                plt.xlabel(param1)
                plt.ylabel(param2)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                if output_dir:
                    plt.savefig(os.path.join(output_dir, 'explored_vs_pending.png'), dpi=300)
                else:
                    plt.show()
                
                plt.close()
        
        # Create a progress over time visualization
        if 'combination_id' in results_df.columns and 'quality_score' in results_df.columns:
            plt.figure(figsize=(12, 6))
            
            # Plot quality score over iterations
            plt.plot(results_df['combination_id'], results_df['quality_score'], 'o-', alpha=0.5)
            
            # Plot best quality score found so far
            best_so_far = results_df['quality_score'].cummax()
            plt.plot(results_df['combination_id'], best_so_far, 'r-', linewidth=2, label='Best score so far')
            
            plt.title('Optimization Progress')
            plt.xlabel('Iteration')
            plt.ylabel('Quality Score')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'optimization_progress.png'), dpi=300)
            else:
                plt.show()
            
            plt.close()
        
        return True
    
    def export_results(self, output_dir):
        """
        Export the current results to CSV and the optimal parameters to JSON.
        
        Args:
            output_dir (str): Directory to save the exports
        
        Returns:
            bool: True if exports were created, False otherwise
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
    
    def propose_additional_exploration(self, output_dir=None, num_suggestions=5):
        """
        Analyze current results and suggest additional parameter combinations to explore.
        This focuses on areas near successful combinations but that haven't been tested yet.
        
        Args:
            output_dir (str, optional): Directory to save suggestions. If None, just displays them.
            num_suggestions (int): Number of additional combinations to suggest
            
        Returns:
            pd.DataFrame: DataFrame with suggested parameter combinations
        """
        if not self.checkpoint_data:
            print("No checkpoint data loaded")
            return pd.DataFrame()
        
        results_df = self.get_results_dataframe()
        if results_df.empty:
            print("No results available to analyze")
            return pd.DataFrame()
        
        # Get parameter ranges
        param_ranges = self.checkpoint_data.get('param_ranges', {})
        if not param_ranges:
            print("No parameter ranges found in checkpoint")
            return pd.DataFrame()
        
        # Focus on numeric parameters
        numeric_params = [
            'community_resolution',
            'min_pattern_frequency',
            'quality_weight_coverage', 
            'quality_weight_redundancy'
        ]
        
        # Filter to parameters that exist in results and have ranges
        numeric_params = [p for p in numeric_params if p in results_df.columns and p in param_ranges]
        
        if not numeric_params:
            print("No numeric parameters found for suggestions")
            return pd.DataFrame()
        
        # Create suggestions based on current results
        suggestions = []
        
        # Strategy 1: Explore around the current best point
        best_idx = results_df['quality_score'].idxmax()
        best_point = results_df.loc[best_idx]
        
        for param in numeric_params:
            # Create variations of the best point by adjusting one parameter at a time
            current_value = best_point[param]
            param_values = param_ranges[param]
            
            if isinstance(param_values, list):
                # Convert all values to the same type for consistent comparisons
                try:
                    if all(isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit()) for v in param_values):
                        # All numeric or numeric strings - convert to float for comparison
                        numeric_values = [float(v) if isinstance(v, str) else v for v in param_values]
                        sorted_values = sorted(numeric_values)
                        current_value_numeric = float(current_value) if isinstance(current_value, str) and current_value.replace('.', '', 1).isdigit() else float(current_value)
                    else:
                        # Mixed types, convert all to strings
                        sorted_values = sorted([str(v) for v in param_values])
                        current_value_numeric = str(current_value)
                    
                    current_idx = sorted_values.index(current_value_numeric)
                    
                    # Try values on either side if they exist
                    for offset in [-2, -1, 1, 2]:
                        new_idx = current_idx + offset
                        if 0 <= new_idx < len(sorted_values):
                            new_value = sorted_values[new_idx]
                            
                            # For parameters that were originally strings but converted to float,
                            # convert back to string to maintain consistency
                            if isinstance(param_values[0], str) and isinstance(new_value, (int, float)):
                                if param_values[0].replace('.', '', 1).isdigit():
                                    # Convert back to string with same format as original
                                    if '.' in param_values[0]:
                                        new_value = str(new_value)
                                    else:
                                        new_value = str(int(new_value))
                            
                            # Create a new suggestion
                            suggestion = {p: best_point[p] for p in results_df.columns 
                                        if p in param_ranges and p != 'combination_id'}
                            suggestion[param] = new_value
                            suggestion['strategy'] = f"Explore around best point ({param})"
                            
                            # Check if this combination has already been evaluated
                            # Create a boolean mask for each parameter
                            mask = pd.Series(True, index=results_df.index)
                            for param_check in numeric_params:
                                if param_check in suggestion and param_check in results_df.columns:
                                    # Handle different types for comparison
                                    if isinstance(suggestion[param_check], (int, float)) and results_df[param_check].dtype == 'object':
                                        # Try to convert strings to numbers for comparison
                                        try:
                                            param_values_converted = results_df[param_check].apply(
                                                lambda x: float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x
                                            )
                                            mask = mask & (param_values_converted == suggestion[param_check])
                                        except:
                                            # If conversion fails, compare as is
                                            mask = mask & (results_df[param_check] == suggestion[param_check])
                                    else:
                                        mask = mask & (results_df[param_check] == suggestion[param_check])
                            
                            # Check if any row matches all conditions
                            if not mask.any():
                                suggestions.append(suggestion)
                except (ValueError, TypeError) as e:
                    print(f"Error processing {param}: {str(e)}")
                    continue
        
        # Strategy 2: Explore high-performing regions
        # Get top performing combinations
        top_n = min(5, len(results_df))
        top_performers = results_df.nlargest(top_n, 'quality_score')
        
        # For each top performer, identify promising dimensions to explore
        for _, performer in top_performers.iterrows():
            # Identify the parameter that most correlates with quality
            # Convert to numeric if possible before calculating correlation
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
            
            for param in numeric_params:
                corr_value = correlation_data.get(param, 0)
                current_value = performer[param]
                param_values = param_ranges[param]
                
                if isinstance(param_values, list):
                    try:
                        # Convert all values to the same type for consistent comparisons
                        if all(isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit()) for v in param_values):
                            # Convert string numbers to float for comparison
                            numeric_values = [float(v) if isinstance(v, str) else v for v in param_values]
                            sorted_values = sorted(numeric_values)
                            current_value_numeric = float(current_value) if isinstance(current_value, str) and current_value.replace('.', '', 1).isdigit() else float(current_value)
                        else:
                            # Mixed types, convert all to strings
                            sorted_values = sorted([str(v) for v in param_values])
                            current_value_numeric = str(current_value)
                        
                        current_idx = sorted_values.index(current_value_numeric)
                        
                        # If positive correlation, try larger values; if negative, try smaller
                        if corr_value > 0 and current_idx < len(sorted_values) - 1:
                            new_value = sorted_values[current_idx + 1]
                        elif corr_value < 0 and current_idx > 0:
                            new_value = sorted_values[current_idx - 1]
                        else:
                            continue
                        
                        # For parameters that were originally strings but converted to float,
                        # convert back to string to maintain consistency
                        if isinstance(param_values[0], str) and isinstance(new_value, (int, float)):
                            if param_values[0].replace('.', '', 1).isdigit():
                                # Convert back to string with same format as original
                                if '.' in param_values[0]:
                                    new_value = str(new_value)
                                else:
                                    new_value = str(int(new_value))
                        
                        # Create a new suggestion
                        suggestion = {p: performer[p] for p in results_df.columns 
                                    if p in param_ranges and p != 'combination_id'}
                        suggestion[param] = new_value
                        suggestion['strategy'] = f"Follow correlation trend ({param})"
                        
                        # Check if this combination has already been evaluated
                        # Create a boolean mask for each parameter
                        mask = pd.Series(True, index=results_df.index)
                        for param_check in numeric_params:
                            if param_check in suggestion and param_check in results_df.columns:
                                # Handle different types for comparison
                                if isinstance(suggestion[param_check], (int, float)) and results_df[param_check].dtype == 'object':
                                    # Try to convert strings to numbers for comparison
                                    try:
                                        param_values_converted = results_df[param_check].apply(
                                            lambda x: float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x
                                        )
                                        mask = mask & (param_values_converted == suggestion[param_check])
                                    except:
                                        # If conversion fails, compare as is
                                        mask = mask & (results_df[param_check] == suggestion[param_check])
                                else:
                                    mask = mask & (results_df[param_check] == suggestion[param_check])
                        
                        # Check if any row matches all conditions
                        if not mask.any():
                            suggestions.append(suggestion)
                    except (ValueError, TypeError):
                        continue
        
        # Strategy 3: Fill gaps in parameter space
        # For each parameter, find values that have been less explored
        for param in numeric_params:
            param_values = param_ranges[param]
            
            if isinstance(param_values, list):
                try:
                    # Count occurrences of each value
                    # Handle type conversion if needed
                    if all(isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit()) for v in param_values):
                        # Convert to numeric for comparison
                        value_counts = results_df[param].astype(float).value_counts()
                        param_values_numeric = [float(v) if isinstance(v, str) else v for v in param_values]
                    else:
                        # Use values as is
                        value_counts = results_df[param].value_counts()
                        param_values_numeric = param_values
                    
                    # Find underexplored values
                    for value in param_values_numeric:
                        if value not in value_counts or value_counts[value] < value_counts.mean() / 2:
                            # This value is underexplored, create combinations with it
                            
                            # Use median values for other parameters
                            suggestion = {}
                            for p in numeric_params:
                                if p == param:
                                    suggestion[p] = value
                                else:
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
                            
                            # Check if this combination has already been evaluated
                            # Create a boolean mask for each parameter
                            mask = pd.Series(True, index=results_df.index)
                            for param_check in numeric_params:
                                if param_check in suggestion and param_check in results_df.columns:
                                    # Handle different types for comparison
                                    if isinstance(suggestion[param_check], (int, float)) and results_df[param_check].dtype == 'object':
                                        # Try to convert strings to numbers for comparison
                                        try:
                                            param_values_converted = results_df[param_check].apply(
                                                lambda x: float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x
                                            )
                                            mask = mask & (param_values_converted == suggestion[param_check])
                                        except:
                                            # If conversion fails, compare as is
                                            mask = mask & (results_df[param_check] == suggestion[param_check])
                                    else:
                                        mask = mask & (results_df[param_check] == suggestion[param_check])
                            
                            # Check if any row matches all conditions
                            if not mask.any():
                                suggestions.append(suggestion)
                except (ValueError, TypeError) as e:
                    print(f"Error processing values for {param}: {str(e)}")
                    continue
        
        # Limit to requested number of suggestions
        suggestions = suggestions[:num_suggestions]
        
        if not suggestions:
            print("Could not generate any new suggestions")
            return pd.DataFrame()
        
        # Convert to DataFrame
        suggestions_df = pd.DataFrame(suggestions)
        
        # Save suggestions if requested
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            suggestions_file = os.path.join(output_dir, "suggested_combinations.csv")
            suggestions_df.to_csv(suggestions_file, index=False)
            print(f"Suggestions saved to {suggestions_file}")
        
        return suggestions_df
    
    def modify_checkpoint(self, add_combinations=None, remove_combinations=None):
        """
        Modify the checkpoint by adding or removing parameter combinations.
        
        Args:
            add_combinations (pd.DataFrame, optional): DataFrame with combinations to add to pending list
            remove_combinations (pd.DataFrame, optional): DataFrame with combinations to remove from pending list
            
        Returns:
            bool: True if the checkpoint was modified, False otherwise
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
            # Get current pending combinations
            pending = self.checkpoint_data.get('pending_combinations', [])
            if not pending:
                print("No pending combinations to remove")
            else:
                # Convert pending to DataFrame for easier comparison
                pending_df = pd.DataFrame(pending)
                
                # Identify combinations to keep
                numeric_params = [
                    'community_resolution',
                    'min_pattern_frequency',
                    'quality_weight_coverage',
                    'quality_weight_redundancy'
                ]
                
                # Filter to parameters that exist in both DataFrames
                common_params = [p for p in numeric_params if p in pending_df.columns and p in remove_combinations.columns]
                
                if not common_params:
                    print("No common parameters found for comparison")
                else:
                    # Find indices of combinations to remove
                    indices_to_remove = []
                    
                    for i, row in enumerate(pending):
                        for _, remove_row in remove_combinations.iterrows():
                            if all(row.get(param) == remove_row[param] for param in common_params if param in row):
                                indices_to_remove.append(i)
                                break
                    
                    # Remove combinations in reverse order to avoid index issues
                    for i in sorted(indices_to_remove, reverse=True):
                        del pending[i]
                    
                    # Update checkpoint data
                    self.checkpoint_data['pending_combinations'] = pending
                    modified = True
                    print(f"Removed {len(indices_to_remove)} combinations from pending list")
        
        # Save modified checkpoint if changes were made
        if modified:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(self.checkpoint_data, f)
            print(f"Modified checkpoint saved to {self.checkpoint_file}")
        
        return modified

def main():
    """
    Main function to manage and analyze checkpoints.
    """
    parser = argparse.ArgumentParser(description='Checkpoint Manager for Parameter Optimization')
    parser.add_argument('checkpoint_file', help='Path to the checkpoint file')
    parser.add_argument('--status', '-s', action='store_true', help='Show search status')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize parameter space coverage')
    parser.add_argument('--export', '-e', metavar='DIR', help='Export results to the specified directory')
    parser.add_argument('--suggest', '-g', metavar='N', type=int, default=0, 
                      help='Suggest N additional parameter combinations to explore')
    parser.add_argument('--output-dir', '-o', default=None, 
                      help='Output directory for visualizations and suggestions')
    
    args = parser.parse_args()
    
    # Create the checkpoint manager
    manager = CheckpointManager(args.checkpoint_file)
    
    # Default output directory
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"checkpoint_analysis_{timestamp}"
    
    # Process commands
    if args.status:
        status = manager.get_status()
        print("\nSearch Status:")
        for key, value in status.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    
    if args.visualize:
        print("\nVisualizing parameter space coverage...")
        manager.visualize_parameter_space_coverage(output_dir)
    
    if args.export:
        print(f"\nExporting results to {args.export}...")
        manager.export_results(args.export)
    
    if args.suggest > 0:
        print(f"\nGenerating {args.suggest} suggested parameter combinations...")
        suggestions = manager.propose_additional_exploration(output_dir, args.suggest)
        
        if not suggestions.empty:
            print("\nSuggested Parameter Combinations:")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(suggestions)

if __name__ == "__main__":
    main()