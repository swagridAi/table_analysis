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

def detect_hierarchical_communities(G, max_level=2, base_resolution=1.0):
    """
    Detect communities hierarchically up to max_level depth.
    
    Args:
        G (networkx.Graph): Network graph to analyze
        max_level (int): Maximum hierarchy depth
        base_resolution (float): Base resolution parameter for the first level
        
    Returns:
        dict: Dictionary with levels as keys and node->community mappings as values
    """
    try:
        import community as community_louvain
    except ImportError:
        print("Warning: python-louvain package not installed. To use hierarchical detection, install with:")
        print("pip install python-louvain")
        return None
    
    from collections import defaultdict
    
    # Create a multi-level dictionary to store communities at each level
    hierarchical_communities = defaultdict(dict)
    
    # Level 0 communities (the original large communities)
    communities_L0 = community_louvain.best_partition(G, resolution=base_resolution)
    
    # Store the level 0 communities
    for node, comm_id in communities_L0.items():
        hierarchical_communities[0][node] = comm_id
    
    # For each level
    for level in range(1, max_level + 1):
        # Group nodes by their community at the previous level
        prev_communities = defaultdict(list)
        for node, comm_id in hierarchical_communities[level-1].items():
            prev_communities[comm_id].append(node)
        
        # For each community at the previous level
        for comm_id, nodes in prev_communities.items():
            # Skip if too small for meaningful subdivision
            if len(nodes) < 5:  # Minimum nodes for subdivision - adjust as needed
                # Just copy the previous level's assignment for these nodes
                for node in nodes:
                    hierarchical_communities[level][node] = f"{comm_id}"
                continue
                
            # Create subgraph for this community
            subgraph = G.subgraph(nodes)
            
            # Apply community detection to this subgraph with higher resolution
            sub_resolution = base_resolution * (level + 1)
            sub_communities = community_louvain.best_partition(subgraph, resolution=sub_resolution)
            
            # Create hierarchical community IDs and store them
            for node, sub_id in sub_communities.items():
                hierarchical_communities[level][node] = f"{comm_id}.{sub_id}"
    
    return hierarchical_communities

def flatten_hierarchical_communities(hierarchical_communities, level):
    """
    Convert hierarchical communities to flat format compatible with existing code.
    
    Args:
        hierarchical_communities (dict): Hierarchical community structure
        level (int): Level to extract
        
    Returns:
        dict: Mapping of nodes to their community IDs at the specified level
    """
    # If the requested level doesn't exist, use the deepest available level
    if level not in hierarchical_communities:
        level = max(hierarchical_communities.keys())
        print(f"Warning: Requested level {level} not found. Using level {level} instead.")
    
    # Convert string IDs to numeric IDs for compatibility
    communities = hierarchical_communities[level]
    unique_ids = sorted(set(communities.values()))
    id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    
    return {node: id_map[comm_id] for node, comm_id in communities.items()}

def analyze_hierarchical_communities(G, hierarchical_communities):
    """
    Analyze hierarchical community structure.
    
    Args:
        G (networkx.Graph): Network graph
        hierarchical_communities (dict): Hierarchical community structure
        
    Returns:
        dict: Analysis results with stats for each level
    """
    results = {}
    
    # Analyze each level
    for level, communities in hierarchical_communities.items():
        # Convert to flat structure for analysis
        flat_communities = flatten_hierarchical_communities(hierarchical_communities, level)
        
        # Use existing function to analyze this level
        level_results = analyze_communities(G, flat_communities)
        
        # Add level-specific information
        level_results['hierarchical_ids'] = set(communities.values())
        level_results['level'] = level
        
        # Store results for this level
        results[level] = level_results
    
    return results