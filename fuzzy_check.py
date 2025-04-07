#!/usr/bin/env python3
"""
Data Co-occurrence Analysis with adaptive fuzzy clustering to handle large communities.
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
import time
from collections import Counter

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

def analyze_community_sizes(primary_communities):
    """Analyze community sizes to identify imbalance issues."""
    community_sizes = Counter(primary_communities.values())
    total_elements = len(primary_communities)
    
    sizes = pd.Series(community_sizes)
    
    print("\nCommunity Size Analysis:")
    print(f"Total communities: {len(community_sizes)}")
    print(f"Largest community: {sizes.max()} elements ({sizes.max()/total_elements:.1%} of total)")
    print(f"Smallest community: {sizes.min()} elements")
    print(f"Average community size: {sizes.mean():.1f} elements")
    
    # Check for imbalance
    size_std = sizes.std()
    size_cv = size_std / sizes.mean()  # Coefficient of variation
    
    imbalance = False
    if size_cv > 0.7:  # Arbitrary threshold for high variation
        imbalance = True
        print("\n⚠️ WARNING: High community size variation detected")
        print(f"Community sizes have high variation (CV={size_cv:.2f})")
    
    if sizes.max() > total_elements * 0.4:  # If largest community has >40% of elements
        imbalance = True
        print("\n⚠️ WARNING: Dominant community detected")
        print(f"Largest community contains {sizes.max()/total_elements:.1%} of all elements")
    
    return {
        "sizes": dict(community_sizes),
        "imbalance_detected": imbalance,
        "size_variation": size_cv,
        "max_percent": sizes.max()/total_elements
    }

def adaptive_fuzzy_clustering(G, max_imbalance=0.35, min_clusters=3, max_clusters=15, 
                             fuzziness_range=(1.5, 2.5), num_attempts=5):
    """
    Perform adaptive fuzzy clustering to avoid large dominant communities.
    
    Args:
        G: NetworkX graph
        max_imbalance: Maximum acceptable size of largest community (as fraction of total)
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
        fuzziness_range: Range of fuzziness values to try
        num_attempts: Number of parameter combinations to try
        
    Returns:
        tuple: (best_primary_communities, best_membership_values, parameters)
    """
    print("\nPerforming adaptive fuzzy clustering to balance community sizes...")
    
    best_result = None
    best_params = None
    best_imbalance = 1.0  # Start with worst possible value
    
    total_elements = G.number_of_nodes()
    
    # Try different parameter combinations
    attempts = 0
    for num_clusters in range(min_clusters, max_clusters + 1):
        # Try more fuzziness values for cluster counts that look promising
        fuzziness_steps = max(2, int(num_attempts / (max_clusters - min_clusters + 1)))
        
        for fuzz_idx in range(fuzziness_steps):
            fuzziness = fuzziness_range[0] + (fuzziness_range[1] - fuzziness_range[0]) * (fuzz_idx / (fuzziness_steps - 1))
            
            attempts += 1
            print(f"  Attempt {attempts}/{num_attempts}: clusters={num_clusters}, fuzziness={fuzziness:.2f}...")
            
            # Run fuzzy clustering with these parameters
            primary_communities, membership_values = detect_communities_fuzzy_cmeans(
                G, num_clusters=num_clusters, fuzziness=fuzziness
            )
            
            # Evaluate balance
            community_sizes = Counter(primary_communities.values())
            largest_community_size = max(community_sizes.values())
            imbalance_metric = largest_community_size / total_elements
            
            print(f"    Result: largest community={largest_community_size} elements ({imbalance_metric:.1%} of total)")
            
            # Check if this is better than current best
            if imbalance_metric < best_imbalance:
                best_imbalance = imbalance_metric
                best_result = (primary_communities, membership_values)
                best_params = {"num_clusters": num_clusters, "fuzziness": fuzziness}
                
                # If we found a good enough result, we can stop
                if imbalance_metric <= max_imbalance:
                    print(f"\n✓ Found acceptable balance with {num_clusters} clusters and fuzziness={fuzziness:.2f}")
                    print(f"  Largest community contains {imbalance_metric:.1%} of elements")
                    return best_result[0], best_result[1], best_params
            
            # Stop if we've reached the maximum number of attempts
            if attempts >= num_attempts:
                break
        
        # Break outer loop too if max attempts reached
        if attempts >= num_attempts:
            break
    
    print(f"\n⚠️ Could not achieve ideal balance ({max_imbalance:.0%}) after {attempts} attempts")
    print(f"Best result: {best_params['num_clusters']} clusters, fuzziness={best_params['fuzziness']:.2f}")
    print(f"Largest community contains {best_imbalance:.1%} of elements")
    
    return best_result[0], best_result[1], best_params

def create_element_clusters_df(membership_df, threshold=0.3):
    """
    Create a more useful representation of element clusters with thresholded memberships.
    
    This function makes it easier to see which elements belong to which communities
    by creating a clean CSV with elements as rows and significant communities as columns.
    """
    # Create a new DataFrame for the output
    elements = membership_df.index.tolist()
    result_data = []
    
    for element in elements:
        # Get memberships for this element
        memberships = membership_df.loc[element]
        
        # Find significant communities (membership > threshold)
        significant = {col: val for col, val in memberships.items() if val >= threshold}
        
        # Sort by membership value (descending)
        sorted_memberships = sorted(significant.items(), key=lambda x: x[1], reverse=True)
        
        # Create a row with primary and secondary communities
        row = {
            'element': element, 
            'table': element.split('.')[1] if '.' in element else '',
            'primary_community': memberships.idxmax(),
            'primary_membership': memberships.max()
        }
        
        # Add significant secondary memberships
        for i, (comm, val) in enumerate(sorted_memberships[1:3], 1):  # Get up to 2 secondary memberships
            if val >= threshold:
                row[f'secondary_community_{i}'] = comm
                row[f'secondary_membership_{i}'] = val
        
        result_data.append(row)
    
    result_df = pd.DataFrame(result_data)
    return result_df

def main():
    """Main function to orchestrate the analysis workflow with adaptive clustering."""
    start_time = time.time()
    print("Starting data co-occurrence analysis with adaptive fuzzy clustering...")
    
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
        
        # Run adaptive fuzzy clustering to find balanced communities
        primary_communities, membership_values, params = adaptive_fuzzy_clustering(
            G, 
            max_imbalance=0.35,  # No community should have more than 35% of elements
            min_clusters=3,
            max_clusters=12,
            num_attempts=8
        )
        
        # Save the parameters used
        params_df = pd.DataFrame([params])
        save_output(params_df, "fuzzy_clustering_parameters.csv",
                   message="Saved clustering parameters")
        
        # Analyze community sizes
        size_analysis = analyze_community_sizes(primary_communities)
        save_output(pd.DataFrame([size_analysis]), "community_size_analysis.csv",
                   message="Saved community size analysis")
        
        # Save primary communities
        primary_df = pd.DataFrame({
            'data_element': list(primary_communities.keys()),
            'primary_community_id': list(primary_communities.values())
        })
        save_output(primary_df, "fuzzy_primary_communities.csv", 
                   message=f"Saved primary community assignments for {len(primary_df)} elements")
        
        # Save full membership matrix
        membership_df = pd.DataFrame.from_dict(membership_values, orient='index')
        membership_df.index.name = 'data_element'
        membership_df.columns = [f'community_{i}' for i in range(membership_df.shape[1])]
        save_output(membership_df, "fuzzy_membership_values.csv", index=True,
                   message=f"Saved full membership values matrix ({membership_df.shape[0]}x{membership_df.shape[1]})")
        
        # Create a more useful representation of communities
        element_clusters_df = create_element_clusters_df(membership_df)
        save_output(element_clusters_df, "element_community_assignments.csv",
                   message="Created user-friendly element-community assignment file")
        
        # Create and save visualization of fuzzy communities
        fuzzy_viz_path = os.path.join(config.VISUALIZATIONS_DIR, "fuzzy_communities_network.png")
        visualize_fuzzy_communities(G, membership_values, output_file=fuzzy_viz_path)
        print(f"Created fuzzy community visualization at {fuzzy_viz_path}")
        
        # Generate community summaries by table
        print("Analyzing community distribution by table...")
        table_community = element_clusters_df.groupby('table')['primary_community'].apply(list).to_dict()
        
        table_analysis = []
        for table, communities in table_community.items():
            community_counts = Counter(communities)
            dominant_community = community_counts.most_common(1)[0][0] if community_counts else None
            dominant_percent = (community_counts.most_common(1)[0][1] / len(communities)) if community_counts else 0
            
            table_analysis.append({
                'table': table,
                'element_count': len(communities),
                'unique_communities': len(community_counts),
                'dominant_community': dominant_community,
                'dominant_community_percent': dominant_percent
            })
        
        save_output(pd.DataFrame(table_analysis), "table_community_analysis.csv",
                   message="Saved table-level community analysis")
        
        elapsed_time = time.time() - start_time
        print(f"\nAnalysis complete! All fuzzy clustering results saved. ({elapsed_time:.2f} seconds)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())