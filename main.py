#!/usr/bin/env python3
"""
Main script for the Data Co-occurrence Analysis project.
This script orchestrates the entire workflow from data loading to visualization.
"""

import os
from src.data.preprocessing import load_data, create_exploded_dataframe
from src.analysis.cooccurrence import calculate_cooccurrence, create_cooccurrence_matrix
from src.visualization.heatmap import create_heatmap
from src.visualization.network import create_network_graph, visualize_network
import networkx as nx
import config

def ensure_directories_exist():
    """Create output directories if they don't exist."""
    for directory in [
        config.PROCESSED_DATA_DIR,
        config.VISUALIZATIONS_DIR,
        config.EXPORTS_DIR
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
    
    # Create and visualize network graph
    print("Creating network graph...")
    G = create_network_graph(co_occurrence, all_elements)
    
    network_file_path = os.path.join(config.VISUALIZATIONS_DIR, "cooccurrence_network.png")
    print(f"Creating network visualization at {network_file_path}...")
    visualize_network(G, output_file=network_file_path)
    
    # Save the graph for further analysis
    graphml_file_path = os.path.join(config.EXPORTS_DIR, "cooccurrence_network.graphml")
    print(f"Saving network graph to {graphml_file_path}...")
    nx.write_graphml(G, graphml_file_path)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()