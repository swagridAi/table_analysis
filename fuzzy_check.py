#!/usr/bin/env python3
"""
Main script for the Data Co-occurrence Analysis project with fuzzy clustering.
"""

import os
import pandas as pd
import numpy as np
from src.data.preprocessing import load_data, create_exploded_dataframe
from src.analysis.cooccurrence import calculate_cooccurrence, create_cooccurrence_matrix
from src.analysis.clustering import detect_communities_fuzzy_cmeans
from src.visualization.heatmap import create_heatmap
from src.visualization.network import create_network_graph, visualize_fuzzy_communities
import networkx as nx
import config
from functools import partial
import time

def ensure_directories_exist():
    """Create output directories if they don't exist."""
    directories = [
        config.PROCESSED_DATA_DIR,
        config.VISUALIZATIONS_DIR,
        config.EXPORTS_DIR,
        config.COMMUNITY_VIZ_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_output(data, filename, directory=None, index=False, message=None):
    """Centralized function to save outputs with proper error handling."""
    directory = directory or config.PROCESSED_DATA_DIR
    filepath = os.path.join(directory, filename)
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=index)
        elif isinstance(data, dict):
            pd.DataFrame.from_dict(data).to_csv(filepath, index=index)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        if message:
            print(message)
        return True
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return False

def create_fuzzy_community_summaries(membership_df, threshold=0.5):
    """Create summary statistics for fuzzy communities - optimized function."""
    # Initialize structure for community summaries
    community_columns = membership_df.columns
    
    # Use vectorized operations instead of loops
    strong_members = {col: membership_df[membership_df[col] > threshold].index.tolist() 
                     for col in community_columns}
    
    summaries = {
        int(col.split('_')[1]): {
            'strong_members_count': len(members),
            'strong_members': ", ".join(members[:10]) + ("..." if len(members) > 10 else ""),
            'weighted_size': membership_df[col].sum(),
            'avg_membership_strength': membership_df[col].mean()
        }
        for col, members in strong_members.items()
    }
    
    return pd.DataFrame.from_dict(summaries, orient='index')

def identify_overlapping_elements(membership_values, threshold=0.3):
    """Identify elements belonging to multiple communities - optimized function."""
    overlap_rows = []

    # Process in batches for large datasets
    for element, memberships in membership_values.items():
        # Use dictionary comprehension for filtering - more efficient
        significant_memberships = {k: v for k, v in memberships.items() if v > threshold}
        
        if len(significant_memberships) > 1:
            overlap_rows.append({
                'data_element': element,
                'overlapping_communities': ",".join(map(str, significant_memberships.keys())),
                'membership_values': ",".join([f"{k}:{v:.3f}" for k, v in significant_memberships.items()])
            })
    
    return pd.DataFrame(overlap_rows) if overlap_rows else None

def main():
    """Main function to orchestrate the analysis workflow."""
    start_time = time.time()
    print("Starting data co-occurrence analysis with fuzzy clustering...")
    
    # Ensure all required directories exist
    ensure_directories_exist()
    
    try:
        # Load the data
        print(f"Loading data from {config.INPUT_FILE}...")
        df = load_data(config.INPUT_FILE)
        
        # Create the exploded dataframe
        print("Processing data elements and reports...")
        df_exploded = create_exploded_dataframe(df)
        save_output(df_exploded, "exploded_data.csv", 
                   message=f"Processed {len(df_exploded)} report-element pairs")
        
        # Calculate co-occurrence
        print("Building co-occurrence relationships...")
        co_occurrence = calculate_cooccurrence(df_exploded)
        print(f"Found {len(co_occurrence)} co-occurrence relationships")
        
        # Create co-occurrence matrix
        cooccurrence_matrix, all_elements = create_cooccurrence_matrix(co_occurrence)
        save_output(cooccurrence_matrix, "cooccurrence_matrix.csv", index=True,
                   message=f"Created {cooccurrence_matrix.shape[0]}x{cooccurrence_matrix.shape[1]} co-occurrence matrix")
        
        # Create heatmap visualization
        heatmap_file_path = os.path.join(config.VISUALIZATIONS_DIR, "cooccurrence_heatmap.png")
        create_heatmap(cooccurrence_matrix, output_file=heatmap_file_path)
        print(f"Created heatmap visualization at {heatmap_file_path}")
        
        # Create network graph 
        print("Building network graph...")
        G = create_network_graph(co_occurrence, all_elements)
        print(f"Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Run fuzzy c-means clustering - with parameters from config if available
        num_clusters = getattr(config, 'FUZZY_CLUSTERS', 8)
        fuzziness = getattr(config, 'FUZZY_FUZZINESS', 2.0)
        print(f"Running fuzzy c-means clustering with {num_clusters} clusters and fuzziness={fuzziness}...")
        
        primary_communities, membership_values = detect_communities_fuzzy_cmeans(
            G, num_clusters=num_clusters, fuzziness=fuzziness
        )
        
        # Process all fuzzy clustering outputs
        print("Processing and saving fuzzy clustering results...")
        
        # Save primary communities
        primary_df = pd.DataFrame({
            'data_element': list(primary_communities.keys()),
            'primary_community_id': list(primary_communities.values())
        })
        save_output(primary_df, "fuzzy_primary_communities.csv", 
                   message=f"Saved primary community assignments for {len(primary_df)} elements")
        
        # Save full membership matrix - convert only once
        membership_df = pd.DataFrame.from_dict(membership_values, orient='index')
        membership_df.index.name = 'data_element'
        membership_df.columns = [f'community_{i}' for i in range(membership_df.shape[1])]
        save_output(membership_df, "fuzzy_membership_values.csv", index=True,
                   message=f"Saved full membership values matrix ({membership_df.shape[0]}x{membership_df.shape[1]})")
        
        # Create and save visualization of fuzzy communities
        fuzzy_viz_path = os.path.join(config.VISUALIZATIONS_DIR, "fuzzy_communities_network.png")
        visualize_fuzzy_communities(G, membership_values, output_file=fuzzy_viz_path)
        print(f"Created fuzzy community visualization at {fuzzy_viz_path}")
        
        # Generate and save community summaries
        summary_df = create_fuzzy_community_summaries(membership_df)
        save_output(summary_df, "fuzzy_community_summary.csv", index=True,
                   message=f"Generated summaries for {len(summary_df)} fuzzy communities")
        
        # Find overlapping elements
        overlap_df = identify_overlapping_elements(membership_values)
        if overlap_df is not None:
            save_output(overlap_df, "fuzzy_overlapping_elements.csv",
                       message=f"Identified {len(overlap_df)} elements with significant community overlap")
        else:
            print("No elements with significant community overlap found")
        
        elapsed_time = time.time() - start_time
        print(f"Analysis complete! All fuzzy clustering results saved. ({elapsed_time:.2f} seconds)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())