#!/usr/bin/env python3
"""
Parameter optimization for Data Co-occurrence Analysis.

This script performs grid search over model parameters to identify optimal configurations
for community detection and product group evaluation. It systematically explores different
parameter combinations, evaluates performance metrics, and visualizes the results to help
determine which parameters most influence model performance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from datetime import datetime

from src.data.preprocessing import load_data, create_exploded_dataframe
from src.analysis.cooccurrence import calculate_cooccurrence, create_cooccurrence_matrix
from src.analysis.clustering import detect_communities, analyze_communities
from src.analysis.metrics import evaluate_product_groups
import config

def ensure_directories_exist(base_dir):
    """Create output directories if they don't exist."""
    for directory in [
        os.path.join(base_dir, "results"),
        os.path.join(base_dir, "visualizations"),
        os.path.join(base_dir, "logs")
    ]:
        os.makedirs(directory, exist_ok=True)
    return os.path.join(base_dir, "results"), os.path.join(base_dir, "visualizations"), os.path.join(base_dir, "logs")

def create_parameter_grid(param_ranges):
    """
    Create a grid of all parameter combinations to explore.
    
    Args:
        param_ranges (dict): Dictionary with parameter names as keys and lists of values to try as values
        
    Returns:
        list: List of dictionaries, each representing a parameter combination
    """
    # Get all parameter names
    param_names = list(param_ranges.keys())
    
    # Generate all combinations of parameter values
    param_values = [param_ranges[name] for name in param_names]
    combinations = list(product(*param_values))
    
    # Convert to list of dictionaries
    grid = []
    for combo in combinations:
        param_dict = {name: value for name, value in zip(param_names, combo)}
        grid.append(param_dict)
    
    return grid

def log_progress(message, log_file):
    """Log progress to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def run_parameter_optimization(input_file, output_base_dir, param_ranges, max_combinations=None):
    """
    Run parameter optimization to find the best configuration.
    
    Args:
        input_file (str): Path to the input CSV file
        output_base_dir (str): Base directory for outputs
        param_ranges (dict): Dictionary with parameter names and ranges to explore
        max_combinations (int, optional): Maximum number of parameter combinations to try
                                         If None, tries all combinations
    
    Returns:
        tuple: (pd.DataFrame with all results, dict with optimal parameters)
    """
    # Create necessary directories
    results_dir, viz_dir, logs_dir = ensure_directories_exist(output_base_dir)
    log_file = os.path.join(logs_dir, f"optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Load and preprocess data (only need to do this once)
    log_progress("Loading and processing data...", log_file)
    df = load_data(input_file)
    df_exploded = create_exploded_dataframe(df)
    
    # Calculate co-occurrence (only need to do this once)
    log_progress("Calculating co-occurrence matrix...", log_file)
    co_occurrence = calculate_cooccurrence(df_exploded)
    cooccurrence_matrix, all_elements = create_cooccurrence_matrix(co_occurrence)
    
    # Get total number of reports for normalization
    report_count = df_exploded['Report'].nunique()
    
    # Create parameter grid
    parameter_grid = create_parameter_grid(param_ranges)
    
    # Limit combinations if requested
    if max_combinations and len(parameter_grid) > max_combinations:
        log_progress(f"Limiting to {max_combinations} out of {len(parameter_grid)} possible combinations", log_file)
        np.random.shuffle(parameter_grid)
        parameter_grid = parameter_grid[:max_combinations]
    
    log_progress(f"Starting parameter optimization with {len(parameter_grid)} combinations", log_file)
    
    # Prepare results storage
    results = []
    
    # Iterate through parameter combinations
    for i, params in enumerate(parameter_grid):
        log_progress(f"Testing combination {i+1}/{len(parameter_grid)}: {params}", log_file)
        
        try:
            # Extract parameters
            community_algorithm = params.get('community_algorithm', config.COMMUNITY_ALGORITHM)
            community_resolution = params.get('community_resolution', config.COMMUNITY_RESOLUTION)
            min_pattern_frequency = params.get('min_pattern_frequency', 2)
            quality_weight_coverage = params.get('quality_weight_coverage', 0.5)
            quality_weight_redundancy = params.get('quality_weight_redundancy', 0.5)
            
            # Detect communities with current parameters
            from networkx import Graph
            G = Graph()
            for (elem1, elem2), count in co_occurrence.items():
                G.add_edge(elem1, elem2, weight=count)
            
            communities = detect_communities(
                G, 
                algorithm=community_algorithm,
                resolution=community_resolution
            )
            
            # Convert communities to product groups
            community_groups = []
            for community_id in set(communities.values()):
                group = [node for node, c_id in communities.items() if c_id == community_id]
                community_groups.append(group)
            
            # Evaluate product groups with current parameters
            metrics = evaluate_product_groups(
                community_groups, 
                cooccurrence_matrix, 
                df_exploded, 
                report_count=report_count,
                min_pattern_frequency=min_pattern_frequency
            )
            
            # Calculate custom quality score if weights are provided
            avg_affinity = np.mean([g["affinity_score"] for g in metrics["group_metrics"]])
            coverage = metrics["overall_metrics"]["weighted_coverage"]
            redundancy = metrics["overall_metrics"]["redundancy_score"]
            
            # Weighted quality score
            quality_score = (avg_affinity * (1-quality_weight_coverage-quality_weight_redundancy) + 
                            coverage * quality_weight_coverage) * (1 - redundancy * quality_weight_redundancy)
            
            # Store results
            result = {
                'combination_id': i,
                'community_algorithm': community_algorithm,
                'community_resolution': community_resolution,
                'min_pattern_frequency': min_pattern_frequency,
                'quality_weight_coverage': quality_weight_coverage,
                'quality_weight_redundancy': quality_weight_redundancy,
                'num_communities': len(community_groups),
                'avg_community_size': np.mean([len(g) for g in community_groups]),
                'avg_affinity_score': avg_affinity,
                'coverage_ratio': metrics["overall_metrics"]["coverage_ratio"],
                'weighted_coverage': coverage,
                'redundancy_score': redundancy,
                'quality_score': quality_score
            }
            
            results.append(result)
            
        except Exception as e:
            log_progress(f"Error with parameters {params}: {str(e)}", log_file)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save all results
    results_file = os.path.join(results_dir, "optimization_results.csv")
    results_df.to_csv(results_file, index=False)
    log_progress(f"Results saved to {results_file}", log_file)
    
    # Find optimal parameters
    if not results_df.empty:
        best_idx = results_df['quality_score'].idxmax()
        optimal_params = results_df.loc[best_idx].to_dict()
        
        log_progress(f"Optimal parameters found: {optimal_params}", log_file)
        
        # Save optimal parameters
        optimal_file = os.path.join(results_dir, "optimal_parameters.csv")
        pd.DataFrame([optimal_params]).to_csv(optimal_file, index=False)
        
        # Create visualizations
        create_parameter_influence_plots(results_df, viz_dir)
    else:
        optimal_params = None
        log_progress("No valid results found.", log_file)
    
    return results_df, optimal_params

def create_parameter_influence_plots(results_df, viz_dir):
    """
    Create visualizations showing how each parameter influences the metrics.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        viz_dir (str): Directory to save visualizations
    """
    if results_df.empty:
        return
    
    # Numeric parameters to analyze
    numeric_params = [
        'community_resolution',
        'min_pattern_frequency',
        'quality_weight_coverage',
        'quality_weight_redundancy'
    ]
    
    # Metrics to analyze
    metrics = [
        'num_communities', 
        'avg_community_size',
        'avg_affinity_score', 
        'coverage_ratio', 
        'redundancy_score', 
        'quality_score'
    ]
    
    # 1. Create parameter influence plots for each numeric parameter
    for param in numeric_params:
        if param not in results_df.columns or results_df[param].nunique() <= 1:
            continue
            
        plt.figure(figsize=(12, 10))
        for i, metric in enumerate(metrics):
            plt.subplot(3, 2, i+1)
            
            # Plot parameter vs metric
            plt.scatter(results_df[param], results_df[metric], alpha=0.7)
            
            # Add trend line if there are enough points
            if len(results_df) > 5:
                try:
                    z = np.polyfit(results_df[param], results_df[metric], 1)
                    p = np.poly1d(z)
                    plt.plot(sorted(results_df[param].unique()), 
                            p(sorted(results_df[param].unique())), 
                            "r--", alpha=0.7)
                except:
                    pass  # Skip trend line if there's an error
            
            plt.title(f"{param} vs {metric}")
            plt.xlabel(param)
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"parameter_influence_{param}.png"), dpi=300)
        plt.close()
    
    # 2. Create algorithm comparison plot
    if 'community_algorithm' in results_df.columns and results_df['community_algorithm'].nunique() > 1:
        plt.figure(figsize=(14, 10))
        
        for i, metric in enumerate(metrics):
            plt.subplot(3, 2, i+1)
            
            # Create box plot for this metric across algorithms
            sns.boxplot(x='community_algorithm', y=metric, data=results_df)
            plt.title(f"Algorithm vs {metric}")
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "algorithm_comparison.png"), dpi=300)
        plt.close()
    
    # 3. Create correlation heatmap between parameters and metrics
    # Combine parameters and metrics
    cols_to_include = [c for c in numeric_params if c in results_df.columns and results_df[c].nunique() > 1] + metrics
    if len(cols_to_include) > 1:  # Need at least 2 columns for correlation
        corr_matrix = results_df[cols_to_include].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Correlation between Parameters and Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "parameter_metric_correlation.png"), dpi=300)
        plt.close()
    
    # 4. Create quality score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['quality_score'], kde=True)
    plt.axvline(results_df['quality_score'].max(), color='r', linestyle='--', 
                label=f'Maximum: {results_df["quality_score"].max():.4f}')
    plt.title("Distribution of Quality Scores")
    plt.xlabel("Quality Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_dir, "quality_score_distribution.png"), dpi=300)
    plt.close()
    
    # 5. Create parallel coordinates plot for top configurations
    top_n = min(10, len(results_df))
    top_configs = results_df.nlargest(top_n, 'quality_score')
    
    from pandas.plotting import parallel_coordinates
    
    # Normalize columns for better visualization
    normalized_cols = numeric_params + ['quality_score']
    normalized_cols = [c for c in normalized_cols if c in top_configs.columns and top_configs[c].nunique() > 1]
    
    if len(normalized_cols) > 1:
        # Create normalized dataframe for plotting
        plot_df = top_configs.copy()
        for col in normalized_cols:
            if plot_df[col].nunique() > 1:  # Only normalize if column has multiple values
                plot_df[col] = (plot_df[col] - plot_df[col].min()) / (plot_df[col].max() - plot_df[col].min())
        
        # Add a class column for the parallel coordinates plot
        plot_df['config_rank'] = [f"Rank {i+1}" for i in range(len(plot_df))]
        
        plt.figure(figsize=(12, 8))
        parallel_coordinates(plot_df, 'config_rank', cols=normalized_cols, colormap='viridis')
        plt.title("Top Configurations Parameter Comparison (Normalized)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "top_configurations_comparison.png"), dpi=300)
        plt.close()

def main():
    """
    Main function to run parameter optimization.
    """
    # Define parameter ranges to explore
    param_ranges = {
        'community_algorithm': ['louvain', 'label_propagation', 'greedy_modularity'],
        'community_resolution': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        'min_pattern_frequency': [2, 3, 4],
        'quality_weight_coverage': [0.3, 0.4, 0.5, 0.6],
        'quality_weight_redundancy': [0.3, 0.4, 0.5, 0.6]
    }
    
    # Define output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.OUTPUT_DIR, f"parameter_optimization_{timestamp}")
    
    # Run optimization
    results_df, optimal_params = run_parameter_optimization(
        input_file=config.INPUT_FILE,
        output_base_dir=output_dir,
        param_ranges=param_ranges,
        max_combinations=50  # Limit combinations to a reasonable number for testing
    )
    
    if optimal_params:
        print("\nOptimal Parameters:")
        for param, value in optimal_params.items():
            if param not in ['combination_id'] and not param.startswith('_'):
                print(f"  {param}: {value}")
    
    print(f"\nAll results and visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()