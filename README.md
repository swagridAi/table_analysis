Data Co-occurrence Analysis
This project analyzes the co-occurrence of data elements across different reports, generating visualizations and network graphs to identify patterns and relationships.
Features

Processes input CSV files containing data elements and their associated reports
Creates an exploded view of data element - report relationships
Calculates co-occurrence matrices to show how frequently data elements appear together
Generates heatmap visualizations of co-occurrence patterns
Creates network graphs showing relationships between data elements
Performs community detection to identify clusters of related data elements
Calculates metrics to assess the viability of product groups
Exports results in various formats for further analysis

Project Structure
data_cooccurrence_analysis/
│
├── data/                          # Data files
│   ├── raw/                       # Original input data
│   │   └── sample_data.csv
│   └── processed/                 # Generated data files
│       ├── exploded_data.csv
│       ├── cooccurrence_matrix.csv
│       ├── communities.csv
│       ├── communities_level0.csv
│       ├── communities_level1.csv
│       └── community_summary.csv
│
├── output/                        # Output files
│   ├── visualizations/            # Visualization outputs
│   │   ├── cooccurrence_heatmap.png
│   │   ├── cooccurrence_network.png
│   │   ├── cooccurrence_network_level1.png
│   │   └── communities/           # Individual community visualizations
│   ├── metrics/                   # Product group metrics
│   │   ├── community_groups_evaluation.xlsx
│   │   ├── table_groups_evaluation.xlsx
│   │   └── custom_groups_evaluation.xlsx
│   ├── metrics_visualizations/    # Metrics visualizations
│   │   ├── strategy_comparison_radar.png
│   │   ├── group_size_affinity_comparison.png
│   │   └── pattern_coverage.png
│   └── exports/                   # Other exports
│       └── cooccurrence_network.graphml
│
├── src/                           # Source code
│   ├── init.py
│   ├── data/                      # Data processing modules
│   │   ├── init.py
│   │   └── preprocessing.py       # Data loading and preprocessing functions
│   │
│   ├── analysis/                  # Analysis modules
│   │   ├── init.py
│   │   ├── cooccurrence.py        # Co-occurrence analysis functions
│   │   ├── clustering.py          # Community detection functions
│   │   └── metrics.py             # Product group evaluation metrics
│   │
│   └── visualization/             # Visualization modules
│       ├── init.py
│       ├── heatmap.py             # Heatmap generation functions
│       ├── network.py             # Network graph functions
│       └── metrics_viz.py         # Metrics visualization functions
│
├── main.py                        # Main script that orchestrates the workflow
├── evaluate_product_groups.py     # Script for evaluating product groups
├── evaluate_with_visualizations.py # Script for evaluating with visualizations
├── config.py                      # Configuration settings
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
Setup

Clone this repository
Create a virtual environment (recommended)
Copypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies
Copypip install -r requirements.txt

Place your input CSV file in the data/raw/ directory
Update config.py if necessary to point to your input file

Input Data Format
The input CSV file should have the following columns:

Data Element Table: The table containing the data element
Data Element Column: The column name of the data element
Enterprise Report Catalog: Reports using the data element (multiple reports separated by newlines)
Critical Data Element: Flag indicating if the data element is critical

Usage
Run the main script to perform the analysis:
Copypython main.py
This will:

Process the input data
Generate the co-occurrence matrix
Create visualizations
Save results to the appropriate output directories

Configuration
You can customize the analysis by modifying settings in config.py, including:

File paths
Visualization parameters (colors, DPI, etc.)
Network graph layout parameters


Community Detection
The project now includes advanced community detection algorithms to identify clusters of data elements that frequently appear together. This helps in:

Discovering natural groupings of related data elements
Identifying functional domains within your data ecosystem
Optimizing report design by understanding data element relationships
Supporting data governance and lineage analysis

Available Algorithms
Three algorithms are available for community detection:

Louvain Method (default):

Optimizes modularity to find communities
Effective for moderate to large networks
Allows fine-tuning community size via resolution parameter
Requires the python-louvain package


Label Propagation:

Fast algorithm suitable for very large networks
Works by spreading labels through the network
Less parameterized and sometimes less stable
Built into NetworkX (no additional dependencies)


Greedy Modularity:

Hierarchical approach to modularity optimization
More computationally intensive but often more accurate
Good for smaller to medium-sized networks
Built into NetworkX (no additional dependencies)

Hierarchical Community Detection
For more granular analysis, the project now supports hierarchical community detection, which:

Identifies sub-communities within larger communities
Creates a multi-level view of data relationships
Allows analysis at different levels of granularity
Uses increasing resolution parameters at deeper levels

The hierarchical approach works by:

First detecting base communities using standard methods
Then recursively analyzing each community to find sub-communities
Creating a hierarchy of communities (e.g., community 0.1.2 means sub-community 2 within sub-community 1 within major community 0)

Community Detection Examples
Example 1: Data Domains
Community detection can reveal natural domains in your data. For example, you might find:

Community 0: Financial data elements (accounts, transactions, balances)
Community 1: Customer profile data elements (demographics, preferences)
Community 2: Product information data elements (inventory, pricing)

Example 2: Hierarchical Structure
Hierarchical detection might reveal:

Community 0: Financial data

Sub-community 0.0: Accounting data
Sub-community 0.1: Transaction data

Sub-community 0.1.0: Retail transactions
Sub-community 0.1.1: Commercial transactions

Configuring Community Detection
You can configure the community detection in config.py:
pythonCopy# Community detection settings
COMMUNITY_ALGORITHM = 'louvain'  # Options: 'louvain', 'label_propagation', 'greedy_modularity'
COMMUNITY_RESOLUTION = 1.0       # Resolution parameter for Louvain method
                                 # Higher values lead to smaller communities
GENERATE_COMMUNITY_SUBGRAPHS = True  # Whether to generate separate visualizations for each community
EXPORT_COMMUNITY_DATA = True      # Whether to export community data to CSV
The COMMUNITY_RESOLUTION parameter (for Louvain method) controls community granularity:

Lower values (0.5-0.8): Produce larger, more inclusive communities
Higher values (1.2+): Produce smaller, more specific communities

Community Outputs
The community detection process generates several outputs:

Color-coded network visualization: Data elements are colored by community membership
Individual community visualizations: Separate graph for each community in output/visualizations/communities/
Community membership data: CSV file mapping data elements to communities in data/processed/communities.csv
Community summary: Statistics about each community in data/processed/community_summary.csv

Interpreting Communities
Communities in the co-occurrence network represent groups of data elements that frequently appear together in reports. These might indicate:

Functional domains: Elements that belong to the same business domain
Report families: Elements that are typically used together in similar reports
Data dependencies: Elements that have business or technical dependencies
Optimization opportunities: Potential for report consolidation or standardization

Product Group Metrics
The project includes functionality to evaluate the viability of different product group strategies using quantitative metrics. This helps in making data-driven decisions about how to organize data elements into optimal groups.
Metrics Overview
Three primary metrics are used to assess product groups:
Affinity Score

Measures how often grouped elements are used together
Higher scores indicate elements that frequently co-occur in reports
Calculated as the average normalized co-occurrence rate between all pairs of elements
Range: 0 to 1, with higher values being better

Coverage Ratio

Measures how well groups reflect common usage patterns
Identifies common element combinations from the actual report data
Calculates the percentage of these patterns fully covered by a single product group
Higher scores indicate better satisfaction of user needs
Range: 0 to 1, with higher values being better

Redundancy Score

Measures unnecessary overlaps between product groups
Counts elements appearing in multiple groups
Lower scores indicate better separation of concerns
Range: 0 to 1, with lower values being better

Quality Score

Combined metric that balances affinity, coverage, and redundancy
Higher scores indicate better overall grouping strategies

Grouping Strategies
The evaluation tools support comparing different product grouping strategies:

Community-based: Groups derived from community detection algorithms
Table-based: Groups based on database table structure
Custom: User-defined groups or alternative grouping strategies

Metrics Visualizations
The metrics visualization module (src/visualization/metrics_viz.py) provides visual representations:

Radar Charts: Compare strategies across all metrics
Group Comparison Charts: Compare group sizes and affinity scores
Overlap Visualizations: Identify redundant elements across groups
Pattern Coverage Charts: Show how well common usage patterns are covered

Using the Metrics
To evaluate product groups:
Copypython evaluate_with_visualizations.py
This will generate:

Excel files with detailed metrics for each grouping strategy
Visualizations comparing the different strategies
Summary statistics showing which approach performs best

The results can be used to:

Identify the optimal grouping strategy for your data
Pinpoint specific improvements needed in your current grouping
Understand which usage patterns are well-covered and which need attention
Detect unnecessary redundancies that could be eliminated

Community Detection Examples
Example 1: Data Domains
Community detection can reveal natural domains in your data. For example, you might find:

Community 1: Financial data elements (accounts, transactions, balances)
Community 2: Customer profile data elements (demographics, preferences)
Community 3: Product information data elements (inventory, pricing)

Output
The analysis generates several outputs:

data/processed/exploded_data.csv: Exploded view of data element - report relationships
data/processed/cooccurrence_matrix.csv: Matrix showing co-occurrence counts
output/visualizations/cooccurrence_heatmap.png: Heatmap visualization
output/visualizations/cooccurrence_network.png: Network graph visualization
output/exports/cooccurrence_network.graphml: GraphML file for further network analysis