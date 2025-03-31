"""
Configuration settings for the Data Co-occurrence Analysis project.
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
EXPORTS_DIR = os.path.join(OUTPUT_DIR, "exports")
COMMUNITY_VIZ_DIR = os.path.join(VISUALIZATIONS_DIR, "communities")

# File paths
INPUT_FILE = os.path.join(RAW_DATA_DIR, "sample_data.csv")

# Visualization settings
HEATMAP_CMAP = "YlGnBu"     # Colormap for heatmap
HEATMAP_DPI = 300           # Resolution for heatmap image
NETWORK_DPI = 300           # Resolution for network graph image

# Network graph settings
NETWORK_LAYOUT_K = 0.5       # Spring layout parameter
NETWORK_ITERATIONS = 50      # Number of iterations for spring layout
NETWORK_LAYOUT_SEED = 42     # Random seed for layout reproducibility
NODE_COLOR = 'skyblue'       # Color of nodes (when not using community colors)
EDGE_COLOR = 'gray'          # Color of edges
NODE_SIZE_MULTIPLIER = 20    # Factor to multiply node importance by for sizing
EDGE_WIDTH_MULTIPLIER = 0.5  # Factor to multiply edge weight by for width

# Standard community detection settings
COMMUNITY_ALGORITHM = 'louvain'  # Options: 'louvain', 'label_propagation', 'greedy_modularity'
COMMUNITY_RESOLUTION = 1.0       # Resolution parameter for Louvain method
                                 # Higher values lead to smaller communities
GENERATE_COMMUNITY_SUBGRAPHS = True  # Whether to generate separate visualizations for each community
EXPORT_COMMUNITY_DATA = True      # Whether to export community data to CSV

# Hierarchical community settings
USE_HIERARCHICAL = True          # Whether to use hierarchical detection
HIERARCHICAL_LEVELS = 2          # Number of levels to detect
HIERARCHICAL_VIZ_LEVEL = 1       # Level to use for visualizations