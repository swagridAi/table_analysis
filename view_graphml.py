from pyvis.network import Network
import networkx as nx
import numpy as np


# Read the GraphML file
G = nx.read_graphml(r"output\exports\cooccurrence_network.graphml")

# Create the network visualization
net = Network(notebook=True, height="750px", width="100%", cdn_resources="remote")

# Calculate layout if positions aren't in the file
if not nx.get_node_attributes(G, 'pos'):
    pos = nx.spring_layout(G, seed=42)
    # Convert numpy arrays to lists for JSON serialization
    pos = {node: list(position) for node, position in pos.items()}
    nx.set_node_attributes(G, pos, 'pos')

# Make sure all node attributes are JSON serializable
for node, attrs in G.nodes(data=True):
    for key, value in list(attrs.items()):
        if isinstance(value, np.ndarray):
            G.nodes[node][key] = value.tolist()

# Make sure all edge attributes are JSON serializable
for u, v, attrs in G.edges(data=True):
    for key, value in list(attrs.items()):
        if isinstance(value, np.ndarray):
            G.edges[u, v][key] = value.tolist()

# Set physics options
net.barnes_hut(spring_length=200, spring_strength=0.01, damping=0.09)

# Convert from NetworkX
net.from_nx(G)

# Additional customization
net.toggle_hide_edges_on_drag(True)
net.toggle_physics(False)  # Turn off physics once initial layout is done

# Save visualization, using notebook=True here since we set it in constructor
net.show("graph.html")