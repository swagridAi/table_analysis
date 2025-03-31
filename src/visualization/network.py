"""
Functions for creating and visualizing network graphs of co-occurrence data.
"""

import matplotlib.pyplot as plt
import networkx as nx
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

def visualize_network(G, output_file=None, dpi=None):
    """
    Visualize the network graph and save to file.
    
    Args:
        G (networkx.Graph): Network graph to visualize
        output_file (str, optional): Path to save the network visualization.
                                     If None, uses default path.
        dpi (int, optional): DPI for the saved image.
                             If None, uses config default.
                             
    Returns:
        matplotlib.figure.Figure: The created figure object if output_file is None
    """
    # Use default values from config if not specified
    dpi = dpi or config.NETWORK_DPI
    
    # Create the figure
    plt.figure(figsize=(14, 12))

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
        iterations=config.NETWORK_ITERATIONS
    )

    # Draw the network
    nx.draw_networkx_nodes(
        G, 
        pos, 
        node_size=node_sizes, 
        node_color=config.NODE_COLOR, 
        alpha=0.8
    )
    
    nx.draw_networkx_edges(
        G, 
        pos, 
        width=edge_weights, 
        alpha=0.5, 
        edge_color=config.EDGE_COLOR
    )
    
    nx.draw_networkx_labels(
        G, 
        pos, 
        font_size=8, 
        font_family='sans-serif'
    )

    plt.title("Data Element Co-occurrence Network")
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure if an output path is provided
    if output_file:
        plt.savefig(output_file, dpi=dpi)
        plt.close()
        return None
    
    # Return the figure if not saving
    return plt.gcf()