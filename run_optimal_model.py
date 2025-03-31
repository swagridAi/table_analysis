#!/usr/bin/env python3
"""
Run the data co-occurrence analysis with optimal parameters.

This script loads the optimal parameters identified through parameter optimization
and runs the full data co-occurrence analysis workflow with those parameters.
"""

import os
import pandas as pd
import json
import sys
from datetime import datetime

from src.data.preprocessing import load_data, create_exploded_dataframe
from src.analysis.cooccurrence import calculate_cooccurrence, create_cooccurrence_matrix
from src.analysis.clustering import detect_communities, analyze_communities
from src.visualization.heatmap import create_heatmap
from src.visualization.network import create_network_graph, visualize_network, visualize_community_subgraphs
import networkx as nx
import config

def ensure_directories_exist(base_dir):
    """Create output directories if they don't exist."""
    for directory in [
        os.path.join(base_dir, "processed"),
        os.path.join(base_dir, "visualizations"),
        os.path.join(base_dir, "exports"),
        os.path.join(base_dir, "visualizations/communities")
    ]:
        os.makedirs(directory, exist_ok=True)

def load_optimal_parameters(params_file):
    """
    Load optimal parameters from file.
    
    Args:
        params_file (str): Path to the parameters file (CSV or JSON)
        
    Returns:
        dict: Dictionary with optimal parameters
    """
    if params_file.endswith('.csv'):
        params_df = pd.read_csv(params_file)
        # Convert first row to dictionary
        params = params_df.iloc[0].to_dict()
    elif params_file.endswith('.json'):
        with open(params_file, 'r') as f:
            params = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {params_file}")
    
    # Filter out metrics and keep only parameters
    param_keys = [
        'community_algorithm', 
        'community_resolution',
        'min_pattern_frequency',
        'quality_weight_coverage',
        'quality_weight_redundancy'
    ]
    
    return {k: params[k] for k in param_keys if k in params}

def run_with_optimal_parameters(params_file, input_file=None, output_dir=None):
    """
    Run the full workflow with optimal parameters.
    
    Args:
        params_file (str): Path to the file with optimal parameters
        input_file (str, optional): Path to the input file (defaults to config.INPUT_FILE)
        output_dir (str, optional): Base directory for output (defaults to timestamp-based directory)
    """
    # Load optimal parameters
    print(f"Loading optimal parameters from {params_file}")
    optimal_params = load_optimal_parameters(params_file)
    print("Optimal parameters:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value}")
    
    # Set up paths
    input_file = input_file or config.INPUT_FILE
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(config.OUTPUT_DIR, f"optimal_run_{timestamp}")
    
    # Create output directories
    ensure_directories_exist(output_dir)
    
    # Save parameters for reference
    params_output = os.path.join(output_dir, "parameters_used.json")
    with open(params_output, 'w') as f:
        json.dump(optimal_params, f, indent=2)
    
    # Define paths
    processed_dir = os.path.join(output_dir, "processed")
    viz_dir = os.path.join(output_dir, "visualizations")
    exports_dir = os.path.join(output_dir, "exports")
    community_viz_dir = os.path.join(viz_dir, "communities")
    
    # Extract parameters with defaults
    community_algorithm = optimal_params.get('community_algorithm', config.COMMUNITY_ALGORITHM)
    community_resolution = optimal_params.get('community_resolution', config.COMMUNITY_RESOLUTION)
    
    # Load the data
    print(f"Loading data from {input_file}...")
    df = load_data(input_file)
    
    # Create the exploded dataframe
    print("Creating exploded dataframe...")
    df_exploded = create_exploded_dataframe(df)
    
    # Save the exploded dataframe
    exploded_file_path = os.path.join(processed_dir, "exploded_data.csv")
    print(f"Saving exploded data to {exploded_file_path}...")
    df_exploded.to_csv(exploded_file_path, index=False)
    
    # Calculate co-occurrence
    print("Calculating co-occurrence matrix...")
    co_occurrence = calculate_cooccurrence(df_exploded)
    
    # Create co-occurrence matrix
    cooccurrence_matrix, all_elements = create_cooccurrence_matrix(co_occurrence)
    
    # Save the matrix
    matrix_file_path = os.path.join(processed_dir, "cooccurrence_matrix.csv")
    print(f"Saving co-occurrence matrix to {matrix_file_path}...")
    cooccurrence_matrix.to_csv(matrix_file_path)
    
    # Create heatmap visualization
    heatmap_file_path = os.path.join(viz_dir, "cooccurrence_heatmap.png")
    print(f"Creating heatmap visualization at {heatmap_file_path}...")
    create_heatmap(cooccurrence_matrix, output_file=heatmap_file_path)
    
    # Create network graph
    print("Creating network graph...")
    G = create_network_graph(co_occurrence, all_elements)
    
    # Detect communities with optimal parameters
    print(f"Detecting communities using {community_algorithm} algorithm with resolution {community_resolution}...")
    communities = detect_communities(
        G, 
        algorithm=community_algorithm,
        resolution=community_resolution
    )
    
    # Analyze communities
    print("Analyzing community structure...")
    community_analysis = analyze_communities(G, communities)
    print(f"Detected {community_analysis['num_communities']} communities")
    
    # Create network visualization with community colors
    network_file_path = os.path.join(viz_dir, "cooccurrence_network.png")
    print(f"Creating network visualization at {network_file_path}...")
    visualize_network(G, output_file=network_file_path, communities=communities)
    
    # Visualize individual communities
    print("Generating individual community visualizations...")
    visualize_community_subgraphs(G, communities, community_viz_dir)
    
    # Export community data to CSV
    print("Exporting community data...")
    # Create a DataFrame with community assignments
    community_df = pd.DataFrame([
        {"data_element": node, "community_id": community_id}
        for node, community_id in communities.items()
    ])
    
    # Save to CSV
    community_file_path = os.path.join(processed_dir, "communities.csv")
    community_df.to_csv(community_file_path, index=False)
    
    # Create a summary DataFrame for communities
    summary_rows = []
    for community_id, elements in community_analysis['community_elements'].items():
        summary_rows.append({
            "community_id": community_id,
            "size": community_analysis['community_sizes'][community_id],
            "density": community_analysis['community_densities'][community_id],
            "elements": ", ".join(elements)
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_file_path = os.path.join(processed_dir, "community_summary.csv")
    summary_df.to_csv(summary_file_path, index=False)
    
    # Save the graph for further analysis
    graphml_file_path = os.path.join(exports_dir, "cooccurrence_network.graphml")
    print(f"Saving network graph to {graphml_file_path}...")
    nx.write_graphml(G, graphml_file_path)
    
    print(f"\nAnalysis with optimal parameters complete!")
    print(f"All results saved to {output_dir}")
    
    return output_dir

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python run_optimal_model.py <parameters_file> [input_file] [output_dir]")
        sys.exit(1)
    
    params_file = sys.argv[1]
    
    input_file = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    run_with_optimal_parameters(params_file, input_file, output_dir)

if __name__ == "__main__":
    main()