"""
Functions for detecting communities and clusters in co-occurrence networks.
"""

import networkx as nx
from collections import defaultdict

def detect_communities(G, algorithm='louvain', resolution=1.0):
    """
    Detect communities in a network graph using various algorithms.
    
    Args:
        G (networkx.Graph): Network graph to analyze
        algorithm (str): Community detection algorithm to use. 
                         Options: 'louvain', 'label_propagation', 'greedy_modularity'
        resolution (float): Resolution parameter for community detection (only for louvain)
                           Higher values lead to smaller communities, lower values to larger ones.
        
    Returns:
        dict: Mapping of nodes to their community IDs
    """
    # Import community detection algorithms
    try:
        # For Louvain method
        import community as community_louvain
    except ImportError:
        print("Warning: python-louvain package not installed. To use Louvain method, install with:")
        print("pip install python-louvain")
        if algorithm == 'louvain':
            algorithm = 'greedy_modularity'
            print("Falling back to greedy_modularity algorithm")
    
    if algorithm == 'louvain':
        # Louvain method (requires python-louvain package)
        return community_louvain.best_partition(G, resolution=resolution)
        
    elif algorithm == 'label_propagation':
        # Label Propagation Algorithm (built into NetworkX)
        communities = nx.algorithms.community.label_propagation_communities(G)
        # Convert to node->community format
        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i
        return community_map
        
    elif algorithm == 'greedy_modularity':
        # Greedy modularity optimization (built into NetworkX)
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        # Convert to node->community format
        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i
        return community_map
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from 'louvain', 'label_propagation', 'greedy_modularity'")

def analyze_communities(G, communities):
    """
    Analyze detected communities to understand their characteristics.
    
    Args:
        G (networkx.Graph): Network graph
        communities (dict): Mapping of nodes to their community IDs
        
    Returns:
        dict: Analysis results including community sizes, internal density, etc.
    """
    # Group nodes by community
    community_groups = defaultdict(list)
    for node, community_id in communities.items():
        community_groups[community_id].append(node)
    
    results = {
        'num_communities': len(community_groups),
        'community_sizes': {},
        'community_densities': {},
        'community_elements': {},
        'modularity': nx.algorithms.community.modularity(G, 
                                                        [community_groups[i] for i in sorted(community_groups.keys())])
    }
    
    # Analyze each community
    for community_id, nodes in community_groups.items():
        # Get the subgraph for this community
        subgraph = G.subgraph(nodes)
        
        # Calculate metrics
        size = len(nodes)
        if size > 1:
            density = nx.density(subgraph)
        else:
            density = 0.0
        
        # Store results
        results['community_sizes'][community_id] = size
        results['community_densities'][community_id] = density
        results['community_elements'][community_id] = nodes
    
    return results