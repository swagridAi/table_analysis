from pyvis.network import Network
import networkx as nx

G = nx.Graph()
G.add_edges_from([("A", "B"), ("B", "C")])

# Option 1: Use notebook=True in constructor (even if not in a notebook)
net = Network(notebook=True)
net.from_nx(G)
net.show("graph.html")