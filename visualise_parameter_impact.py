#!/usr/bin/env python3
"""
Visualize the impact of individual parameters on model performance.

This script creates interactive visualizations to explore parameter-performance relationships,
allowing for deep analysis of which parameters most strongly influence model outcomes.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from datetime import datetime
import argparse

def load_results(results_file):
    """
    Load parameter optimization results from file.
    
    Args:
        results_file (str): Path to the CSV file with optimization results
        
    Returns:
        pd.DataFrame: DataFrame with results
    """
    return pd.read_csv(results_file)

def create_parameter_impact_visualizations(results_df, output_dir):
    """
    Create rich visualizations showing parameter impacts on performance metrics.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Parameter Impact Matrix
    create_parameter_impact_matrix(results_df, output_dir)
    
    # 2. Detailed Parameter Response Curves
    create_parameter_response_curves(results_df, output_dir)
    
    # 3. Parameter Interaction Heatmaps
    create_parameter_interaction_heatmaps(results_df, output_dir)
    
    # 4. Performance Distribution Analysis
    create_performance_distribution_analysis(results_df, output_dir)
    
    # 5. Overall Parameter Sensitivity Analysis
    create_parameter_sensitivity_analysis(results_df, output_dir)

def create_parameter_impact_matrix(results_df, output_dir):
    """
    Create a matrix showing the impact of each parameter on each metric.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        output_dir (str): Directory to save visualizations
    """
    # Define parameters and metrics to analyze
    params = [
        'community_algorithm', 
        'community_resolution', 
        'min_pattern_frequency',
        'quality_weight_coverage', 
        'quality_weight_redundancy'
    ]
    
    # Filter to parameters that actually vary in the results
    params = [p for p in params if p in results_df.columns and results_df[p].nunique() > 1]
    
    metrics = [
        'num_communities', 
        'avg_community_size',
        'avg_affinity_score', 
        'coverage_ratio', 
        'weighted_coverage',
        'redundancy_score', 
        'quality_score'
    ]
    
    # Filter to metrics that exist in the results
    metrics = [m for m in metrics if m in results_df.columns]
    
    if not params or not metrics:
        print("Not enough parameter or metric variation for impact matrix")
        return
    
    # Calculate correlations between parameters and metrics
    correlation_data = []
    
    for param in params:
        param_type = results_df[param].dtype
        
        for metric in metrics:
            if param_type == 'object':  # Categorical parameter
                # Calculate eta squared (effect size) for categorical parameters
                # Group by the parameter and calculate mean and variance of the metric
                groups = results_df.groupby(param)[metric]
                grand_mean = results_df[metric].mean()
                n = len(results_df)
                
                # Between group sum of squares
                ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for _, group in groups)
                
                # Total sum of squares
                ss_total = sum((results_df[metric] - grand_mean)**2)
                
                # Calculate eta squared (effect size)
                if ss_total == 0:
                    eta_squared = 0
                else:
                    eta_squared = ss_between / ss_total
                
                impact_value = eta_squared
                impact_type = 'eta_squared'
            else:  # Numeric parameter
                # Calculate Spearman correlation (more robust to outliers and non-linear relationships)
                correlation = results_df[[param, metric]].corr(method='spearman').iloc[0, 1]
                impact_value = correlation
                impact_type = 'correlation'
            
            correlation_data.append({
                'parameter': param,
                'metric': metric,
                'impact_value': impact_value,
                'impact_type': impact_type
            })
    
    impact_df = pd.DataFrame(correlation_data)
    
    # Create a pivot table for visualization
    impact_pivot = impact_df.pivot(index='parameter', columns='metric', values='impact_value')
    
    # Create custom diverging colormap (red for negative, white for zero, blue for positive)
    colors = [(0.8, 0.2, 0.2), (1, 1, 1), (0.2, 0.2, 0.8)]
    cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(impact_pivot, annot=True, cmap=cmap, center=0, vmin=-1, vmax=1, fmt='.2f')
    plt.title('Parameter Impact on Performance Metrics', fontsize=15)
    plt.ylabel('Parameter', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    
    # Rotate y-axis labels for better readability
    plt.yticks(rotation=0)
    
    # Add a note about the impact type
    impact_types = impact_df['impact_type'].unique()
    if len(impact_types) == 1:
        if impact_types[0] == 'correlation':
            note = "Values show Spearman correlation coefficients"
        else:
            note = "Values show eta squared (effect size)"
    else:
        note = "Values show Spearman correlation or eta squared (for categorical parameters)"
    
    plt.figtext(0.5, 0.01, note, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_impact_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the impact data for reference
    impact_df.to_csv(os.path.join(output_dir, 'parameter_impact_data.csv'), index=False)

def create_parameter_response_curves(results_df, output_dir):
    """
    Create detailed response curves showing how each parameter affects each metric.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        output_dir (str): Directory to save visualizations
    """
    # Define numeric parameters and metrics
    numeric_params = [
        'community_resolution', 
        'min_pattern_frequency',
        'quality_weight_coverage', 
        'quality_weight_redundancy'
    ]
    
    # Filter to parameters that exist and vary
    numeric_params = [p for p in numeric_params if p in results_df.columns and results_df[p].nunique() > 1]
    
    # Key metrics to analyze
    key_metrics = ['avg_affinity_score', 'coverage_ratio', 'redundancy_score', 'quality_score']
    
    # Filter to metrics that exist
    key_metrics = [m for m in key_metrics if m in results_df.columns]
    
    if not numeric_params or not key_metrics:
        print("Not enough parameter or metric variation for response curves")
        return
    
    # Create a response curve for each numeric parameter
    for param in numeric_params:
        plt.figure(figsize=(14, 10))
        
        for i, metric in enumerate(key_metrics):
            plt.subplot(2, 2, i+1)
            
            # Scatter plot of raw data
            sns.scatterplot(x=param, y=metric, data=results_df, alpha=0.7)
            
            # Add a trend line using LOWESS smoothing
            try:
                sns.regplot(x=param, y=metric, data=results_df, scatter=False, 
                          lowess=True, line_kws={'color': 'red', 'lw': 2})
            except:
                # Fall back to linear trend if LOWESS fails
                sns.regplot(x=param, y=metric, data=results_df, scatter=False,
                          line_kws={'color': 'red', 'lw': 2})
            
            plt.title(f"Effect of {param} on {metric}")
            plt.xlabel(param)
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
            
            # Add a vertical line at the optimal value
            best_idx = results_df['quality_score'].idxmax()
            optimal_value = results_df.loc[best_idx, param]
            plt.axvline(x=optimal_value, color='green', linestyle='--', 
                      label=f'Optimal: {optimal_value:.3f}')
            
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'response_curve_{param}.png'), dpi=300)
        plt.close()
    
    # Create a special visualization for algorithm comparison if it exists and varies
    if 'community_algorithm' in results_df.columns and results_df['community_algorithm'].nunique() > 1:
        plt.figure(figsize=(14, 10))
        
        for i, metric in enumerate(key_metrics):
            plt.subplot(2, 2, i+1)
            
            # Box plot for categorical comparison
            sns.boxplot(x='community_algorithm', y=metric, data=results_df)
            
            # Add individual points
            sns.stripplot(x='community_algorithm', y=metric, data=results_df, 
                         color='black', alpha=0.5, jitter=True)
            
            plt.title(f"Effect of Algorithm on {metric}")
            plt.xlabel('Algorithm')
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'algorithm_comparison.png'), dpi=300)
        plt.close()

def create_parameter_interaction_heatmaps(results_df, output_dir):
    """
    Create heatmaps showing how parameter interactions affect quality score.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        output_dir (str): Directory to save visualizations
    """
    # Define numeric parameters to analyze
    numeric_params = [
        'community_resolution', 
        'min_pattern_frequency',
        'quality_weight_coverage', 
        'quality_weight_redundancy'
    ]
    
    # Filter to parameters that exist and vary
    numeric_params = [p for p in numeric_params if p in results_df.columns and results_df[p].nunique() > 1]
    
    if len(numeric_params) < 2:
        print("Not enough varying numeric parameters for interaction analysis")
        return
    
    # Create interaction heatmaps for pairs of parameters
    for i, param1 in enumerate(numeric_params):
        for param2 in numeric_params[i+1:]:
            # Create pivot table
            try:
                # Create bins for the parameters
                param1_bins = pd.cut(results_df[param1], bins=5)
                param2_bins = pd.cut(results_df[param2], bins=5)
                
                # Create pivot table with binned parameters
                pivot = results_df.pivot_table(
                    values='quality_score',
                    index=param1_bins,
                    columns=param2_bins,
                    aggfunc='mean'
                )
                
                # Create heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
                plt.title(f'Interaction Effect of {param1} and {param2} on Quality Score', fontsize=13)
                plt.xlabel(param2, fontsize=12)
                plt.ylabel(param1, fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'interaction_{param1}_{param2}.png'), dpi=300)
                plt.close()
            except Exception as e:
                print(f"Could not create interaction heatmap for {param1} and {param2}: {str(e)}")
                continue

def create_performance_distribution_analysis(results_df, output_dir):
    """
    Create visualizations showing the distribution of performance metrics.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        output_dir (str): Directory to save visualizations
    """
    # Key metrics to analyze
    metrics = ['avg_affinity_score', 'coverage_ratio', 'redundancy_score', 'quality_score']
    
    # Filter to metrics that exist
    metrics = [m for m in metrics if m in results_df.columns]
    
    if not metrics:
        print("No metrics found for distribution analysis")
        return
    
    # Create violin plots for each metric
    plt.figure(figsize=(14, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        # Violin plot with embedded box plot
        sns.violinplot(y=results_df[metric], inner='box')
        
        # Add individual points
        sns.stripplot(y=results_df[metric], color='black', alpha=0.5, jitter=True)
        
        # Mark the optimal value
        best_idx = results_df['quality_score'].idxmax()
        optimal_value = results_df.loc[best_idx, metric]
        plt.axhline(y=optimal_value, color='red', linestyle='--', 
                  label=f'Optimal: {optimal_value:.3f}')
        
        plt.title(f"Distribution of {metric}")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_distributions.png'), dpi=300)
    plt.close()
    
    # Create a parallel coordinates plot of the top configurations
    try:
        from pandas.plotting import parallel_coordinates
        
        # Get top configurations
        top_n = min(10, len(results_df))
        top_configs = results_df.nlargest(top_n, 'quality_score').copy()
        
        # Normalize metrics for better visualization
        for metric in metrics:
            if top_configs[metric].nunique() > 1:
                min_val = top_configs[metric].min()
                max_val = top_configs[metric].max()
                if max_val > min_val:
                    top_configs[f'{metric}_norm'] = (top_configs[metric] - min_val) / (max_val - min_val)
                else:
                    top_configs[f'{metric}_norm'] = 0.5  # Constant value if all are the same
        
        # Add rank column
        top_configs['rank'] = [f'Rank {i+1}' for i in range(len(top_configs))]
        
        # Create parallel coordinates plot
        plt.figure(figsize=(12, 8))
        parallel_coordinates(top_configs, 'rank', [f'{m}_norm' for m in metrics], colormap='viridis')
        plt.title('Performance Profile of Top Configurations')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_configurations_performance.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not create parallel coordinates plot: {str(e)}")

def create_parameter_sensitivity_analysis(results_df, output_dir):
    """
    Create a radar chart showing the overall sensitivity of metrics to parameters.
    
    Args:
        results_df (pd.DataFrame): DataFrame with optimization results
        output_dir (str): Directory to save visualizations
    """
    # Define parameters and metrics
    params = [
        'community_algorithm', 
        'community_resolution', 
        'min_pattern_frequency',
        'quality_weight_coverage', 
        'quality_weight_redundancy'
    ]
    
    # Filter to parameters that exist and vary
    params = [p for p in params if p in results_df.columns and results_df[p].nunique() > 1]
    
    metrics = ['avg_affinity_score', 'coverage_ratio', 'redundancy_score', 'quality_score']
    
    # Filter to metrics that exist
    metrics = [m for m in metrics if m in results_df.columns]
    
    if not params or not metrics:
        print("Not enough parameter or metric variation for sensitivity analysis")
        return
    
    # Calculate sensitivity scores
    sensitivity_data = []
    
    for param in params:
        param_sensitivities = {}
        
        if results_df[param].dtype == 'object':  # Categorical parameter
            # Use eta squared for categorical parameters
            for metric in metrics:
                groups = results_df.groupby(param)[metric]
                grand_mean = results_df[metric].mean()
                
                # Between group sum of squares
                ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for _, group in groups)
                
                # Total sum of squares
                ss_total = sum((results_df[metric] - grand_mean)**2)
                
                # Calculate eta squared (effect size)
                if ss_total == 0:
                    eta_squared = 0
                else:
                    eta_squared = ss_between / ss_total
                
                param_sensitivities[metric] = abs(eta_squared)  # Use absolute value for sensitivity
        else:  # Numeric parameter
            # Use correlation analysis for numeric parameters
            for metric in metrics:
                correlation = results_df[[param, metric]].corr(method='spearman').iloc[0, 1]
                param_sensitivities[metric] = abs(correlation)  # Use absolute value for sensitivity
        
        # Calculate average sensitivity across all metrics
        avg_sensitivity = np.mean(list(param_sensitivities.values()))
        
        # Store results
        sensitivity_data.append({
            'parameter': param,
            'average_sensitivity': avg_sensitivity,
            **param_sensitivities
        })
    
    sensitivity_df = pd.DataFrame(sensitivity_data)
    
    # Create radar chart
    categories = ['average_sensitivity'] + metrics
    N = len(categories)
    
    # Create angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, fontsize=12)
    
    # Draw parameter lines
    for i, row in sensitivity_df.iterrows():
        values = [row[cat] for cat in categories]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['parameter'])
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Parameter Sensitivity Analysis', size=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_sensitivity_radar.png'), dpi=300)
    plt.close()
    
    # Create a bar chart of average sensitivity
    plt.figure(figsize=(10, 6))
    sns.barplot(x='parameter', y='average_sensitivity', data=sensitivity_df.sort_values('average_sensitivity', ascending=False))
    plt.title('Overall Parameter Sensitivity Ranking')
    plt.xlabel('Parameter')
    plt.ylabel('Average Sensitivity')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_sensitivity_ranking.png'), dpi=300)
    plt.close()
    
    # Save sensitivity data
    sensitivity_df.to_csv(os.path.join(output_dir, 'parameter_sensitivity_data.csv'), index=False)

def main():
    """Main function to run parameter impact visualization."""
    parser = argparse.ArgumentParser(description='Visualize parameter impact on model performance')
    parser.add_argument('results_file', help='Path to the parameter optimization results CSV file')
    parser.add_argument('--output-dir', '-o', default=None, 
                      help='Directory to save visualizations (default: timestamp-based directory)')
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"parameter_impact_visualizations_{timestamp}"
    else:
        output_dir = args.output_dir
    
    # Load results
    results_df = load_results(args.results_file)
    
    # Create visualizations
    create_parameter_impact_visualizations(results_df, output_dir)
    
    print(f"Parameter impact visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()