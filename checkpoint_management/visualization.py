"""
Visualization functions for parameter optimization analysis.

This module contains functions for creating visualizations of parameter space exploration,
optimization progress, and parameter relationships.
"""

import os
from typing import Dict, List, Optional, Any, Union, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .param_utils import is_numeric_convertible, convert_to_numeric_if_possible, get_numeric_params


def visualize_parameter_space(
    results_df: pd.DataFrame,
    param_ranges: Dict[str, List[Any]],
    output_dir: Optional[str] = None
) -> bool:
    """
    Create visualizations of parameter space exploration.
    
    Args:
        results_df: DataFrame with optimization results
        param_ranges: Dictionary with parameter ranges
        output_dir: Directory to save visualizations
        
    Returns:
        True if visualizations were created, False otherwise
    """
    if results_df.empty:
        print("No results available to visualize")
        return False
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine which parameters to visualize
    numeric_params = get_numeric_params(results_df)
    
    if len(numeric_params) < 2:
        print("Not enough numeric parameters to visualize parameter space")
        return False
    
    # Create visualizations
    create_parameter_pair_scatterplots(results_df, numeric_params, param_ranges, output_dir)
    
    if len(numeric_params) >= 3:
        create_3d_parameter_visualization(results_df, numeric_params, output_dir)
    
    create_progress_visualization(results_df, output_dir)
    
    return True


def visualize_explored_vs_pending(
    results_df: pd.DataFrame,
    pending_df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> bool:
    """
    Create visualization comparing explored and pending parameter combinations.
    
    Args:
        results_df: DataFrame with results
        pending_df: DataFrame with pending combinations
        output_dir: Directory to save visualization
        
    Returns:
        True if visualization was created, False otherwise
    """
    if results_df.empty or pending_df.empty:
        return False
    
    # Determine which parameters to visualize
    numeric_params = get_numeric_params(results_df)
    
    if len(numeric_params) < 2:
        return False
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if parameters exist in pending dataframe
    param1, param2 = numeric_params[:2]
    if param1 not in pending_df.columns or param2 not in pending_df.columns:
        return False
    
    create_explored_vs_pending_visualization(
        results_df, pending_df, param1, param2, output_dir
    )
    
    return True


def create_parameter_pair_scatterplots(
    results_df: pd.DataFrame,
    numeric_params: List[str],
    param_ranges: Dict[str, List[Any]],
    output_dir: Optional[str]
) -> None:
    """
    Create scatterplots for pairs of parameters.
    
    Args:
        results_df: DataFrame with results
        numeric_params: List of numeric parameters
        param_ranges: Dictionary with parameter ranges
        output_dir: Directory to save visualizations
    """
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
            if param1 in param_ranges and param2 in param_ranges:
                add_parameter_space_boundaries(
                    plt, param_ranges[param1], param_ranges[param2]
                )
            
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


def add_parameter_space_boundaries(
    plt_obj: plt, 
    x_range: List[Any], 
    y_range: List[Any]
) -> None:
    """
    Add parameter space boundaries to a plot.
    
    Args:
        plt_obj: Matplotlib plot object
        x_range: Range of x values
        y_range: Range of y values
    """
    if not isinstance(x_range, list) or not isinstance(y_range, list):
        return
        
    try:
        # Convert to numeric if possible
        x_values = convert_to_numeric_if_possible(x_range)
        y_values = convert_to_numeric_if_possible(y_range)
        
        # Only proceed if we have numeric values
        if all(isinstance(x, (int, float)) for x in x_values) and all(isinstance(y, (int, float)) for y in y_values):
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
            
            # Add slightly transparent background to show full parameter space
            plt_obj.fill_betweenx([y_min, y_max], x_min, x_max, alpha=0.1, color='gray')
            
            # Set axis limits to parameter space with some padding
            x_padding = 0.1 * (x_max - x_min)
            y_padding = 0.1 * (y_max - y_min)
            plt_obj.xlim(x_min - x_padding, x_max + x_padding)
            plt_obj.ylim(y_min - y_padding, y_max + y_padding)
    except (TypeError, ValueError):
        # If conversion fails, skip parameter space boundaries
        pass


def create_3d_parameter_visualization(
    results_df: pd.DataFrame,
    numeric_params: List[str],
    output_dir: Optional[str]
) -> None:
    """
    Create a 3D visualization of parameter space.
    
    Args:
        results_df: DataFrame with results
        numeric_params: List of numeric parameters
        output_dir: Directory to save visualization
    """
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


def create_explored_vs_pending_visualization(
    results_df: pd.DataFrame,
    pending_df: pd.DataFrame,
    param1: str,
    param2: str,
    output_dir: Optional[str]
) -> None:
    """
    Create visualization comparing explored vs pending parameter combinations.
    
    Args:
        results_df: DataFrame with results
        pending_df: DataFrame with pending combinations
        param1: First parameter to plot
        param2: Second parameter to plot
        output_dir: Directory to save visualization
    """
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


def create_progress_visualization(
    results_df: pd.DataFrame,
    output_dir: Optional[str]
) -> None:
    """
    Create visualization showing optimization progress over time.
    
    Args:
        results_df: DataFrame with results
        output_dir: Directory to save visualization
    """
    if 'combination_id' not in results_df.columns or 'quality_score' not in results_df.columns:
        return
        
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