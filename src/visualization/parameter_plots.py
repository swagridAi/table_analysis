"""
Visualization utilities for parameter optimization results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_parameter_influence_plots(results_df, output_dir, logger=None):
    """
    Create visualizations showing how each parameter influences the metrics.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        output_dir (str): Directory to save visualizations
        logger (callable, optional): Function to log messages
    """
    def log(message):
        """Helper function to handle logging."""
        if logger:
            logger(message)
        else:
            print(message)
            
    if results_df.empty:
        log("No results to visualize.")
        return
    
    # Prepare data and get parameters and metrics to analyze
    numeric_params, metrics = _prepare_visualization_data(results_df, log)
    
    # Create individual visualization types
    _create_parameter_metric_plots(results_df, numeric_params, metrics, output_dir)
    _create_algorithm_comparison_plots(results_df, metrics, output_dir)
    _create_correlation_heatmap(results_df, numeric_params, metrics, output_dir)
    _create_quality_score_distribution(results_df, output_dir)
    
    log(f"Parameter influence visualizations saved to {output_dir}")

def _prepare_visualization_data(results_df, log_fn):
    """
    Prepare data for visualization by identifying parameters and metrics to analyze.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        log_fn (callable): Function to log messages
        
    Returns:
        tuple: (numeric_params, metrics) lists of parameters and metrics to analyze
    """
    # Numeric parameters to analyze
    numeric_params = [
        'community_resolution',
        'min_pattern_frequency',
        'quality_weight_coverage',
        'quality_weight_redundancy'
    ]
    
    # Filter to parameters that exist and vary
    numeric_params = [p for p in numeric_params if p in results_df.columns and results_df[p].nunique() > 1]

    # Debug parameter types
    _debug_parameter_types(results_df, numeric_params, log_fn)
    
    # Metrics to analyze
    metrics = [
        'num_communities', 
        'avg_community_size',
        'avg_affinity_score', 
        'coverage_ratio', 
        'redundancy_score', 
        'quality_score'
    ]
    
    # Filter to metrics that exist
    metrics = [m for m in metrics if m in results_df.columns]
    
    return numeric_params, metrics

def _debug_parameter_types(results_df, numeric_params, log_fn):
    """
    Debug the types of parameters to identify potential issues.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        numeric_params (list): List of numeric parameters to debug
        log_fn (callable): Function to log messages
    """
    for param in numeric_params:
        values = results_df[param].tolist()
        log_fn(f"\nValues for parameter {param}:")
        for i, v in enumerate(values[:5]):  # Show first 5 values
            log_fn(f"  Value {i}: '{v}' of type {type(v)}")
        
        try:
            # Try to do operations that might trigger type errors
            results_df[param].sort_values()
        except TypeError as e:
            # Define a simple debug_type_info function if not available
            def debug_type_info(value1, value2, context=""):
                log_fn(f"\n*** TYPE COMPARISON DEBUG ({context}) ***")
                log_fn(f"Value 1: '{value1}' of type {type(value1)}")
                log_fn(f"Value 2: '{value2}' of type {type(value2)}")
                log_fn("*** END DEBUG ***\n")
                
            debug_type_info(values[0], values[1], f"sorting parameter {param}")

def _create_parameter_metric_plots(results_df, numeric_params, metrics, output_dir):
    """
    Create scatter plots showing the relationship between each parameter and metric.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        numeric_params (list): List of numeric parameters to analyze
        metrics (list): List of metrics to analyze
        output_dir (str): Directory to save visualizations
    """
    for param in numeric_params:
        plt.figure(figsize=(12, 10))
        
        for i, metric in enumerate(metrics):
            _create_single_parameter_metric_plot(
                results_df, param, metric, i+1
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"parameter_influence_{param}.png"), dpi=300)
        plt.close()

def _create_single_parameter_metric_plot(results_df, param, metric, subplot_idx):
    """
    Create a single scatter plot for a parameter-metric pair.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        param (str): Parameter name
        metric (str): Metric name
        subplot_idx (int): Subplot index
    """
    plt.subplot(3, 2, subplot_idx)
    
    # Plot parameter vs metric
    plt.scatter(results_df[param], results_df[metric], alpha=0.7)
    
    # Add trend line if there are enough points
    _add_trend_line(results_df, param, metric)
    
    plt.title(f"{param} vs {metric}")
    plt.xlabel(param)
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)

def _add_trend_line(results_df, param, metric):
    """
    Add a trend line to a parameter-metric plot if there are enough data points.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        param (str): Parameter name
        metric (str): Metric name
    """
    if len(results_df) > 5:
        try:
            z = np.polyfit(results_df[param], results_df[metric], 1)
            p = np.poly1d(z)
            plt.plot(sorted(results_df[param].unique()), 
                     p(sorted(results_df[param].unique())), 
                     "r--", alpha=0.7)
        except:
            pass  # Skip trend line if there's an error

def _create_algorithm_comparison_plots(results_df, metrics, output_dir):
    """
    Create box plots comparing metrics across different community algorithms.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        metrics (list): List of metrics to analyze
        output_dir (str): Directory to save visualizations
    """
    if 'community_algorithm' in results_df.columns and results_df['community_algorithm'].nunique() > 1:
        plt.figure(figsize=(14, 10))
        
        for i, metric in enumerate(metrics):
            plt.subplot(3, 2, i+1)
            
            # Create box plot for this metric across algorithms
            sns.boxplot(x='community_algorithm', y=metric, data=results_df)
            plt.title(f"Algorithm vs {metric}")
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "algorithm_comparison.png"), dpi=300)
        plt.close()

def _create_correlation_heatmap(results_df, numeric_params, metrics, output_dir):
    """
    Create a correlation heatmap between parameters and metrics.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        numeric_params (list): List of numeric parameters
        metrics (list): List of metrics
        output_dir (str): Directory to save visualizations
    """
    # Combine parameters and metrics
    cols_to_include = [c for c in numeric_params if c in results_df.columns and results_df[c].nunique() > 1] + metrics
    
    if len(cols_to_include) > 1:  # Need at least 2 columns for correlation
        corr_matrix = results_df[cols_to_include].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Correlation between Parameters and Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "parameter_metric_correlation.png"), dpi=300)
        plt.close()

def _create_quality_score_distribution(results_df, output_dir):
    """
    Create a histogram showing the distribution of quality scores.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        output_dir (str): Directory to save visualizations
    """
    if 'quality_score' in results_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['quality_score'], kde=True)
        plt.axvline(results_df['quality_score'].max(), color='r', linestyle='--', 
                    label=f'Maximum: {results_df["quality_score"].max():.4f}')
        plt.title("Distribution of Quality Scores")
        plt.xlabel("Quality Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "quality_score_distribution.png"), dpi=300)
        plt.close()