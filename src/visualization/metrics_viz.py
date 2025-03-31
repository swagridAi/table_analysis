"""
Visualization functions for product group metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def create_metrics_radar_chart(metrics_list, strategy_names, output_file=None, dpi=300):
    """
    Create a radar chart comparing different product grouping strategies.
    
    Args:
        metrics_list (list): List of metrics dictionaries from evaluate_product_groups
        strategy_names (list): List of names for each strategy
        output_file (str, optional): Path to save the visualization
        dpi (int, optional): DPI for the saved image
    
    Returns:
        matplotlib.figure.Figure: The created figure object if output_file is None
    """
    # Extract metrics for comparison
    metrics_to_compare = [
        "affinity_score",
        "coverage_ratio",
        "quality_score",
        "inverse_redundancy"  # We'll invert redundancy so higher is better for all metrics
    ]
    
    labels = [
        "Affinity Score",
        "Coverage Ratio",
        "Quality Score",
        "Low Redundancy"
    ]
    
    # Prepare data
    values = []
    for metrics in metrics_list:
        overall = metrics["overall_metrics"]
        avg_affinity = np.mean([g["affinity_score"] for g in metrics["group_metrics"]])
        
        # Invert redundancy so higher is better (1 - redundancy)
        inverse_redundancy = 1 - overall["redundancy_score"]
        
        values.append([
            avg_affinity,
            overall["coverage_ratio"],
            overall["quality_score"],
            inverse_redundancy
        ])
    
    # Normalize values to 0-1 range for fair comparison
    values_array = np.array(values)
    min_vals = np.min(values_array, axis=0)
    max_vals = np.max(values_array, axis=0)
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    
    values_normalized = (values_array - min_vals) / range_vals
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Number of variables
    N = len(metrics_to_compare)
    
    # Angle of each axis
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Plot each strategy
    for i, strategy in enumerate(strategy_names):
        values_for_plot = values_normalized[i].tolist()
        values_for_plot += values_for_plot[:1]  # Close the loop
        
        ax.plot(angles, values_for_plot, linewidth=2, label=strategy)
        ax.fill(angles, values_for_plot, alpha=0.1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Add legend
    plt.legend(loc='upper right')
    
    plt.title("Comparison of Product Grouping Strategies", size=15)
    plt.tight_layout()
    
    # Save the figure if an output path is provided
    if output_file:
        plt.savefig(output_file, dpi=dpi)
        plt.close()
        return None
    
    # Return the figure if not saving
    return fig

def create_group_comparison_chart(metrics_list, strategy_names, output_file=None, dpi=300):
    """
    Create bar charts comparing group sizes and affinities across strategies.
    
    Args:
        metrics_list (list): List of metrics dictionaries from evaluate_product_groups
        strategy_names (list): List of names for each strategy
        output_file (str, optional): Path to save the visualization
        dpi (int, optional): DPI for the saved image
    
    Returns:
        matplotlib.figure.Figure: The created figure object if output_file is None
    """
    # Prepare data for plotting
    data = []
    
    for i, metrics in enumerate(metrics_list):
        for group in metrics["group_metrics"]:
            data.append({
                "Strategy": strategy_names[i],
                "Group ID": f"{strategy_names[i]} - {group['group_id']}",
                "Size": group["size"],
                "Affinity Score": group["affinity_score"]
            })
    
    df = pd.DataFrame(data)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot group sizes
    sns.barplot(x="Strategy", y="Size", data=df, ax=ax1, estimator=np.mean, errorbar=None)
    ax1.set_title("Average Group Size by Strategy")
    ax1.set_ylabel("Average Number of Elements")
    
    # Add error bars for standard deviation
    for i, strategy in enumerate(strategy_names):
        strategy_data = df[df["Strategy"] == strategy]["Size"]
        ax1.errorbar(i, strategy_data.mean(), yerr=strategy_data.std(), fmt='o', color='black')
    
    # Plot affinity scores
    sns.barplot(x="Strategy", y="Affinity Score", data=df, ax=ax2)
    ax2.set_title("Average Affinity Score by Strategy")
    ax2.set_ylabel("Average Affinity Score")
    
    # Add error bars for standard deviation
    for i, strategy in enumerate(strategy_names):
        strategy_data = df[df["Strategy"] == strategy]["Affinity Score"]
        ax2.errorbar(i, strategy_data.mean(), yerr=strategy_data.std(), fmt='o', color='black')
    
    plt.tight_layout()
    
    # Save the figure if an output path is provided
    if output_file:
        plt.savefig(output_file, dpi=dpi)
        plt.close()
        return None
    
    # Return the figure if not saving
    return fig

def visualize_group_overlap(metrics, output_file=None, dpi=300, max_elements=30):
    """
    Create a heatmap showing element overlap between groups.
    
    Args:
        metrics (dict): Metrics dictionary from evaluate_product_groups
        output_file (str, optional): Path to save the visualization
        dpi (int, optional): DPI for the saved image
        max_elements (int): Maximum number of elements to show
    
    Returns:
        matplotlib.figure.Figure: The created figure object if output_file is None
    """
    # Get groups and elements
    groups = metrics["group_metrics"]
    
    # Filter to most duplicated elements first
    element_counts = metrics["element_redundancy"]
    sorted_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Limit to top elements if there are too many
    elements_to_show = [e[0] for e in sorted_elements[:max_elements]]
    
    # Create overlap matrix
    overlap_matrix = np.zeros((len(groups), len(elements_to_show)))
    
    for i, group in enumerate(groups):
        group_elements = set(group["elements"])
        for j, element in enumerate(elements_to_show):
            overlap_matrix[i, j] = 1 if element in group_elements else 0
    
    # Create dataframe for better labeling
    overlap_df = pd.DataFrame(
        overlap_matrix, 
        index=[f"Group {g['group_id']}" for g in groups],
        columns=[e.split('.')[-1] for e in elements_to_show]  # Just show column names for readability
    )
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(overlap_df, cmap="Blues", cbar=False)
    
    plt.title("Element Overlap Between Groups")
    plt.tight_layout()
    
    # Save the figure if an output path is provided
    if output_file:
        plt.savefig(output_file, dpi=dpi)
        plt.close()
        return None
    
    # Return the figure if not saving
    return plt.gcf()

def visualize_pattern_coverage(metrics, output_file=None, dpi=300, top_n=20):
    """
    Visualize how well the groups cover common usage patterns.
    
    Args:
        metrics (dict): Metrics dictionary from evaluate_product_groups
        output_file (str, optional): Path to save the visualization
        dpi (int, optional): DPI for the saved image
        top_n (int): Show only top N patterns by frequency
    
    Returns:
        matplotlib.figure.Figure: The created figure object if output_file is None
    """
    # Extract pattern coverage data
    pattern_coverage = metrics["pattern_coverage"]
    
    # Convert to DataFrame
    data = []
    for pattern, details in pattern_coverage.items():
        data.append({
            "Pattern": pattern[:40] + "..." if len(pattern) > 40 else pattern,  # Truncate for display
            "Frequency": details["frequency"],
            "Covered": "Yes" if details["covered"] else "No"
        })
    
    df = pd.DataFrame(data)
    
    # Sort by frequency and take top N
    df = df.sort_values("Frequency", ascending=False).head(top_n)
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    
    # Use different colors for covered vs. not covered
    sns.barplot(x="Frequency", y="Pattern", hue="Covered", data=df, palette={"Yes": "green", "No": "red"})
    
    plt.title(f"Coverage of Top {top_n} Common Usage Patterns")
    plt.xlabel("Pattern Frequency (Number of Reports)")
    plt.tight_layout()
    
    # Save the figure if an output path is provided
    if output_file:
        plt.savefig(output_file, dpi=dpi)
        plt.close()
        return None
    
    # Return the figure if not saving
    return plt.gcf()

def generate_all_visualizations(metrics_list, strategy_names, output_dir):
    """
    Generate all visualizations for the metrics.
    
    Args:
        metrics_list (list): List of metrics dictionaries from evaluate_product_groups
        strategy_names (list): List of names for each strategy
        output_dir (str): Directory to save the visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison radar chart
    radar_file = os.path.join(output_dir, "strategy_comparison_radar.png")
    create_metrics_radar_chart(metrics_list, strategy_names, radar_file)
    
    # Create group comparison charts
    comparison_file = os.path.join(output_dir, "group_size_affinity_comparison.png")
    create_group_comparison_chart(metrics_list, strategy_names, comparison_file)
    
    # Create overlap visualization for each strategy
    for i, metrics in enumerate(metrics_list):
        overlap_file = os.path.join(output_dir, f"{strategy_names[i]}_group_overlap.png")
        visualize_group_overlap(metrics, overlap_file)
        
        # Create pattern coverage visualization for each strategy
        pattern_file = os.path.join(output_dir, f"{strategy_names[i]}_pattern_coverage.png")
        visualize_pattern_coverage(metrics, pattern_file)
    
    print(f"All visualizations saved to {output_dir}")