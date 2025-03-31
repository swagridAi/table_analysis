"""
Functions for creating and visualizing network graphs of co-occurrence data.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import config

def create_network_graph(co_occurrence, all_elements):
    """
    Create a network graph from co-occurrence data.
    
    Args:
        co_occurrence (dict): Dictionary with (element1, element2) tuples as keys
                              and co-occurrence counts as values
        all_elements (list): List of all unique data elements
        
    Returns:
        networkx.Graph: Network graph with nodes as data elements and edges weighted by co-occurrence
    """
    G = nx.Graph()
    
    # Add all nodes
    for element in all_elements:
        G.add_node(element)
    
    # Add edges with weights based on co-occurrence
    for (elem1, elem2), count in co_occurrence.items():
        G.add_edge(elem1, elem2, weight=count)
    
    return G

def visualize_network(G, output_file=None, dpi=None, communities=None):
    """
    Visualize the network graph and save to file.
    
    Args:
        G (networkx.Graph): Network graph to visualize
        output_file (str, optional): Path to save the network visualization.
                                     If None, uses default path.
        dpi (int, optional): DPI for the saved image.
                             If None, uses config default.
        communities (dict, optional): Mapping of nodes to their community IDs.
                                     If provided, nodes will be colored by community.
                             
    Returns:
        matplotlib.figure.Figure: The created figure object if output_file is None
    """
    # Use default values from config if not specified
    dpi = dpi or config.NETWORK_DPI
    
    # Create the figure
    plt.figure(figsize=(14, 12))

    # Create the figure with a proper layout for colorbar
    fig, ax = plt.subplots(figsize=(14, 12))

    # Use co-occurrence count for edge width and node size based on total connections
    edge_weights = [
        G[u][v]['weight'] * config.EDGE_WIDTH_MULTIPLIER 
        for u, v in G.edges()
    ]
    
    node_sizes = [
        sum([G[node][neighbor]['weight'] for neighbor in G[node]]) * config.NODE_SIZE_MULTIPLIER 
        for node in G.nodes()
    ]

    # Position nodes using force-directed layout
    pos = nx.spring_layout(
        G, 
        k=config.NETWORK_LAYOUT_K, 
        iterations=config.NETWORK_ITERATIONS,
        seed=config.NETWORK_LAYOUT_SEED if hasattr(config, 'NETWORK_LAYOUT_SEED') else None
    )

    # Draw nodes colored by community if communities are provided
    if communities:
        # Generate a color map based on community
        community_values = [communities.get(node, 0) for node in G.nodes()]
        num_communities = len(set(communities.values()))
        
        # Use a qualitative colormap that can handle many communities
        if num_communities <= 10:
            cmap = plt.cm.get_cmap('tab10')
        else:
            cmap = plt.cm.get_cmap('rainbow', num_communities)
            
        # Draw nodes with community colors
        nodes = nx.draw_networkx_nodes(
            G, 
            pos, 
            ax=ax,
            node_size=node_sizes, 
            node_color=community_values,
            cmap=cmap,
            alpha=0.8
        )
        
        # Add a color bar to show community mapping
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(community_values)))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Community")
        
    else:
        # Draw with default color if no communities provided
        nx.draw_networkx_nodes(
            G, 
            pos, 
            ax=ax,
            node_size=node_sizes, 
            node_color=config.NODE_COLOR, 
            alpha=0.8
        )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, 
        pos, 
        ax=ax,
        width=edge_weights, 
        alpha=0.5, 
        edge_color=config.EDGE_COLOR
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, 
        pos, 
        ax=ax,
        font_size=8, 
        font_family='sans-serif'
    )

    ax.set_title("Data Element Co-occurrence Network")
    ax.axis('off')
    fig.tight_layout()
    
    # Save the figure if an output path is provided
    if output_file:
        plt.savefig(output_file, dpi=dpi)
        plt.close()
        return None
    
    # Return the figure if not saving
    return plt.gcf()

def visualize_community_subgraphs(G, communities, output_dir, prefix="community_", dpi=None):
    """
    Create separate visualizations for each community subgraph.
    
    Args:
        G (networkx.Graph): Network graph
        communities (dict): Mapping of nodes to their community IDs
        output_dir (str): Directory to save the community visualizations
        prefix (str): Prefix for the output filenames
        dpi (int, optional): DPI for the saved images
    """
    # Use default DPI from config if not specified
    dpi = dpi or config.NETWORK_DPI
    
    # Group nodes by community
    community_groups = {}
    for node, community_id in communities.items():
        if community_id not in community_groups:
            community_groups[community_id] = []
        community_groups[community_id].append(node)
    
    # Visualize each community subgraph
    for community_id, nodes in community_groups.items():
        # Skip if fewer than 2 nodes
        if len(nodes) < 2:
            continue
            
        # Get the subgraph for this community
        subgraph = G.subgraph(nodes)
        
        # Output filename
        output_file = f"{output_dir}/{prefix}{community_id}.png"
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
                # Create figure with explicit axes
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate node sizes based on weights
        node_sizes = [
            sum([subgraph[node][neighbor]['weight'] for neighbor in subgraph[node]]) 
            * config.NODE_SIZE_MULTIPLIER 
            for node in subgraph.nodes()
        ]
        
        # Calculate edge widths
        edge_weights = [
            subgraph[u][v]['weight'] * config.EDGE_WIDTH_MULTIPLIER 
            for u, v in subgraph.edges()
        ]
        
        # Position nodes
        pos = nx.spring_layout(
            subgraph, 
            k=config.NETWORK_LAYOUT_K * 1.5,  # Use larger k for better spacing in subgraphs
            iterations=config.NETWORK_ITERATIONS,
            seed=config.NETWORK_LAYOUT_SEED if hasattr(config, 'NETWORK_LAYOUT_SEED') else None
        )
        
        # Draw nodes, edges, and labels with explicit axes
        nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_size=node_sizes, node_color=config.NODE_COLOR, alpha=0.8)
        nx.draw_networkx_edges(subgraph, pos, ax=ax, width=edge_weights, alpha=0.6, edge_color=config.EDGE_COLOR)
        nx.draw_networkx_labels(subgraph, pos, ax=ax, font_size=10, font_family='sans-serif')
        
        ax.set_title(f"Community {community_id} (Size: {len(nodes)})")
        ax.axis('off')
        fig.tight_layout()
        
        # Save the figure
        plt.savefig(output_file, dpi=dpi)
        plt.close()