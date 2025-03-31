#!/usr/bin/env python3
"""
Script to evaluate and visualize product groups using co-occurrence metrics.
"""

import os
import pandas as pd
import numpy as np
from src.data.preprocessing import load_data, create_exploded_dataframe
from src.analysis.cooccurrence import calculate_cooccurrence, create_cooccurrence_matrix
from src.analysis.metrics import evaluate_product_groups
from src.visualization.metrics_viz import generate_all_visualizations
import config

def load_community_based_groups(community_file):
    """
    Load product groups based on community detection results.
    
    Args:
        community_file (str): Path to communities.csv file
        
    Returns:
        list: List of lists, where each inner list contains the data elements for a product group
    """
    community_df = pd.read_csv(community_file)
    
    # Group by community_id and collect data elements
    product_groups = []
    for community_id, group_df in community_df.groupby('community_id'):
        product_groups.append(group_df['data_element'].tolist())
    
    return product_groups

def create_table_based_groups(df_exploded):
    """
    Create product groups based on database tables.
    
    Args:
        df_exploded (pandas.DataFrame): Exploded dataframe with data elements
        
    Returns:
        list: List of lists, where each inner list contains the data elements for a table-based product group
    """
    # Extract table name from data_element (assuming format "table_name.column_name")
    df_exploded['table'] = df_exploded['data_element'].apply(lambda x: x.split('.')[0])
    
    # Group by table
    product_groups = []
    for table, group_df in df_exploded.groupby('table'):
        elements = group_df['data_element'].unique().tolist()
        product_groups.append(elements)
    
    return product_groups

def create_custom_groups(df_exploded):
    """
    Create custom product groups for demonstration.
    
    Args:
        df_exploded (pandas.DataFrame): Exploded dataframe with data elements
        
    Returns:
        list: List of custom product groups
    """
    # Get all unique data elements
    all_elements = df_exploded['data_element'].unique().tolist()
    
    # For demonstration, create 5 groups with some overlapping elements
    np.random.seed(42)  # For reproducibility
    
    num_groups = 5
    elements_per_group = len(all_elements) // num_groups
    
    custom_groups = []
    for i in range(num_groups):
        # Select random elements for this group
        start_idx = i * elements_per_group
        end_idx = start_idx + elements_per_group + np.random.randint(0, 5)  # Add some variation
        end_idx = min(end_idx, len(all_elements))
        
        # Add some overlap between groups
        if i > 0:
            overlap_size = np.random.randint(1, 5)
            group = all_elements[start_idx-overlap_size:end_idx]
        else:
            group = all_elements[start_idx:end_idx]
            
        custom_groups.append(group)
    
    return custom_groups

def save_evaluation_results(metrics, output_file):
    """
    Save evaluation results to a CSV file.
    
    Args:
        metrics (dict): Metrics dictionary from evaluate_product_groups
        output_file (str): Path to save the CSV file
    """
    # Extract group metrics
    group_metrics_df = pd.DataFrame(metrics["group_metrics"])
    group_metrics_df['elements'] = group_metrics_df['elements'].apply(lambda x: ', '.join(x))
    
    # Extract overall metrics
    overall_metrics_df = pd.DataFrame([metrics["overall_metrics"]])
    
    # Create pattern coverage dataframe
    pattern_coverage = []
    for pattern, details in metrics["pattern_coverage"].items():
        pattern_coverage.append({
            "pattern": pattern,
            "frequency": details["frequency"],
            "covered": details["covered"]
        })
    pattern_coverage_df = pd.DataFrame(pattern_coverage)
    
    # Create element redundancy dataframe
    element_redundancy = []
    for element, count in metrics["element_redundancy"].items():
        element_redundancy.append({
            "element": element,
            "group_count": count
        })
    element_redundancy_df = pd.DataFrame(element_redundancy)
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(output_file) as writer:
        overall_metrics_df.to_excel(writer, sheet_name='Overall Metrics', index=False)
        group_metrics_df.to_excel(writer, sheet_name='Group Metrics', index=False)
        pattern_coverage_df.to_excel(writer, sheet_name='Pattern Coverage', index=False)
        element_redundancy_df.to_excel(writer, sheet_name='Element Redundancy', index=False)
    
    print(f"Evaluation results saved to {output_file}")

def main():
    """
    Main function to evaluate different product grouping strategies.
    """
    # Load and preprocess data
    print("Loading and processing data...")
    df = load_data(config.INPUT_FILE)
    df_exploded = create_exploded_dataframe(df)
    
    # Calculate co-occurrence
    print("Calculating co-occurrence matrix...")
    co_occurrence = calculate_cooccurrence(df_exploded)
    cooccurrence_matrix, all_elements = create_cooccurrence_matrix(co_occurrence)
    
    # Get total number of reports for normalization
    report_count = df_exploded['Report'].nunique()
    
    # Create output directories if they don't exist
    metrics_dir = os.path.join(config.OUTPUT_DIR, "metrics")
    viz_dir = os.path.join(config.OUTPUT_DIR, "metrics_visualizations")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # For storing all metrics for comparison
    all_metrics = []
    strategy_names = []
    
    # Evaluate community-based product groups
    print("\nEvaluating community-based product groups...")
    community_file = os.path.join(config.PROCESSED_DATA_DIR, "communities.csv")
    if os.path.exists(community_file):
        community_groups = load_community_based_groups(community_file)
        community_metrics = evaluate_product_groups(
            community_groups, 
            cooccurrence_matrix, 
            df_exploded, 
            report_count=report_count
        )
        
        # Save results
        community_output = os.path.join(metrics_dir, "community_groups_evaluation.xlsx")
        save_evaluation_results(community_metrics, community_output)
        
        # Store for comparison
        all_metrics.append(community_metrics)
        strategy_names.append("Community-based")
        
        # Display summary
        print(f"Community-based groups: {len(community_groups)} groups")
        print(f"Average Affinity Score: {np.mean([g['affinity_score'] for g in community_metrics['group_metrics']]):.4f}")
        print(f"Coverage Ratio: {community_metrics['overall_metrics']['coverage_ratio']:.4f}")
        print(f"Redundancy Score: {community_metrics['overall_metrics']['redundancy_score']:.4f}")
        print(f"Quality Score: {community_metrics['overall_metrics']['quality_score']:.4f}")
    else:
        print(f"Community file not found: {community_file}")
    
    # Evaluate table-based product groups
    print("\nEvaluating table-based product groups...")
    table_groups = create_table_based_groups(df_exploded)
    table_metrics = evaluate_product_groups(
        table_groups, 
        cooccurrence_matrix, 
        df_exploded, 
        report_count=report_count
    )
    
    # Save results
    table_output = os.path.join(metrics_dir, "table_groups_evaluation.xlsx")
    save_evaluation_results(table_metrics, table_output)
    
    # Store for comparison
    all_metrics.append(table_metrics)
    strategy_names.append("Table-based")
    
    # Display summary
    print(f"Table-based groups: {len(table_groups)} groups")
    print(f"Average Affinity Score: {np.mean([g['affinity_score'] for g in table_metrics['group_metrics']]):.4f}")
    print(f"Coverage Ratio: {table_metrics['overall_metrics']['coverage_ratio']:.4f}")
    print(f"Redundancy Score: {table_metrics['overall_metrics']['redundancy_score']:.4f}")
    print(f"Quality Score: {table_metrics['overall_metrics']['quality_score']:.4f}")
    
    # Evaluate custom product groups
    print("\nEvaluating custom product groups...")
    custom_groups = create_custom_groups(df_exploded)
    custom_metrics = evaluate_product_groups(
        custom_groups, 
        cooccurrence_matrix, 
        df_exploded, 
        report_count=report_count
    )
    
    # Save results
    custom_output = os.path.join(metrics_dir, "custom_groups_evaluation.xlsx")
    save_evaluation_results(custom_metrics, custom_output)
    
    # Store for comparison
    all_metrics.append(custom_metrics)
    strategy_names.append("Custom")
    
    # Display summary
    print(f"Custom groups: {len(custom_groups)} groups")
    print(f"Average Affinity Score: {np.mean([g['affinity_score'] for g in custom_metrics['group_metrics']]):.4f}")
    print(f"Coverage Ratio: {custom_metrics['overall_metrics']['coverage_ratio']:.4f}")
    print(f"Redundancy Score: {custom_metrics['overall_metrics']['redundancy_score']:.4f}")
    print(f"Quality Score: {custom_metrics['overall_metrics']['quality_score']:.4f}")
    
    # Compare strategies
    print("\nComparison of Grouping Strategies:")
    strategies = []
    for i, name in enumerate(strategy_names):
        quality_score = all_metrics[i]["overall_metrics"]["quality_score"]
        strategies.append((name, quality_score))
    
    for name, score in sorted(strategies, key=lambda x: x[1], reverse=True):
        print(f"{name}: Quality Score = {score:.4f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_all_visualizations(all_metrics, strategy_names, viz_dir)
    
    print("\nAnalysis complete!")
    print("Results saved to:", metrics_dir)
    print("Visualizations saved to:", viz_dir)

if __name__ == "__main__":
    main()