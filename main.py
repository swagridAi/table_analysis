#!/usr/bin/env python3
"""
Main script for the Data Co-occurrence Analysis project.
This script orchestrates the entire workflow from data loading to visualization.
"""

import os
import pandas as pd
from src.data.preprocessing import load_data, create_exploded_dataframe
from src.analysis.cooccurrence import calculate_cooccurrence, create_cooccurrence_matrix
from src.analysis.clustering import detect_communities, analyze_communities
from src.visualization.heatmap import create_heatmap
from src.visualization.network import create_network_graph, visualize_network, visualize_community_subgraphs
import networkx as nx
import config

def ensure_directories_exist():
    """Create output directories if they don't exist."""
    for directory in [
        config.PROCESSED_DATA_DIR,
        config.VISUALIZATIONS_DIR,
        config.EXPORTS_DIR,
        config.COMMUNITY_VIZ_DIR
    ]:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main function to orchestrate the analysis workflow."""
    # Ensure all required directories exist
    ensure_directories_exist()
    
    # Load the data
    print(f"Loading data from {config.INPUT_FILE}...")
    df = load_data(config.INPUT_FILE)
    
    # Create the exploded dataframe
    print("Creating exploded dataframe...")
    df_exploded = create_exploded_dataframe(df)
    
    # Save the exploded dataframe
    exploded_file_path = os.path.join(config.PROCESSED_DATA_DIR, "exploded_data.csv")
    print(f"Saving exploded data to {exploded_file_path}...")
    df_exploded.to_csv(exploded_file_path, index=False)
    
    # Calculate co-occurrence
    print("Calculating co-occurrence matrix...")
    co_occurrence = calculate_cooccurrence(df_exploded)
    
    # Create co-occurrence matrix
    cooccurrence_matrix, all_elements = create_cooccurrence_matrix(co_occurrence)
    
    # Save the matrix
    matrix_file_path = os.path.join(config.PROCESSED_DATA_DIR, "cooccurrence_matrix.csv")
    print(f"Saving co-occurrence matrix to {matrix_file_path}...")
    cooccurrence_matrix.to_csv(matrix_file_path)
    
    # Create heatmap visualization
    heatmap_file_path = os.path.join(config.VISUALIZATIONS_DIR, "cooccurrence_heatmap.png")
    print(f"Creating heatmap visualization at {heatmap_file_path}...")
    create_heatmap(cooccurrence_matrix, output_file=heatmap_file_path)
    
    # Create network graph
    print("Creating network graph...")
    G = create_network_graph(co_occurrence, all_elements)
    
    # Detect communities
    print(f"Detecting communities using {config.COMMUNITY_ALGORITHM} algorithm...")
    communities = detect_communities(
        G, 
        algorithm=config.COMMUNITY_ALGORITHM,
        resolution=config.COMMUNITY_RESOLUTION
    )
    
    # Analyze communities
    print("Analyzing community structure...")
    community_analysis = analyze_communities(G, communities)
    print(f"Detected {community_analysis['num_communities']} communities")
    
    # Print community information
    for community_id, size in sorted(community_analysis['community_sizes'].items()):
        density = community_analysis['community_densities'][community_id]
        print(f"Community {community_id}: {size} elements, density: {density:.4f}")
    
    # Create network visualization with community colors
    network_file_path = os.path.join(config.VISUALIZATIONS_DIR, "cooccurrence_network.png")
    print(f"Creating network visualization at {network_file_path}...")
    visualize_network(G, output_file=network_file_path, communities=communities)
    
    # Create separate visualizations for each community if configured
    if config.GENERATE_COMMUNITY_SUBGRAPHS:
        print("Generating individual community visualizations...")
        visualize_community_subgraphs(G, communities, config.COMMUNITY_VIZ_DIR)
    
    # Export community data to CSV if configured
    if config.EXPORT_COMMUNITY_DATA:
        print("Exporting community data...")
        # Create a DataFrame with community assignments
        community_df = pd.DataFrame([
            {"data_element": node, "community_id": community_id}
            for node, community_id in communities.items()
        ])
        
        # Save to CSV
        community_file_path = os.path.join(config.PROCESSED_DATA_DIR, "communities.csv")
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
        summary_file_path = os.path.join(config.PROCESSED_DATA_DIR, "community_summary.csv")
        summary_df.to_csv(summary_file_path, index=False)
    
    # Save the graph for further analysis
    graphml_file_path = os.path.join(config.EXPORTS_DIR, "cooccurrence_network.graphml")
    print(f"Saving network graph to {graphml_file_path}...")
    nx.write_graphml(G, graphml_file_path)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()