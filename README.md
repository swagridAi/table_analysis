Data Co-occurrence Analysis
This project analyzes the co-occurrence of data elements across different reports, generating visualizations and network graphs to identify patterns and relationships.
Features

Processes input CSV files containing data elements and their associated reports
Creates an exploded view of data element - report relationships
Calculates co-occurrence matrices to show how frequently data elements appear together
Generates heatmap visualizations of co-occurrence patterns
Creates network graphs showing relationships between data elements
Exports results in various formats for further analysis

Project Structure
Copydata_cooccurrence_analysis/
│
├── data/                          # Data files
│   ├── raw/                       # Original input data
│   │   └── sample_data.csv
│   └── processed/                 # Generated data files
│       ├── exploded_data.csv
│       └── cooccurrence_matrix.csv
│
├── output/                        # Output files
│   ├── visualizations/            # Visualization outputs
│   │   ├── cooccurrence_heatmap.png
│   │   └── cooccurrence_network.png
│   └── exports/                   # Other exports
│       └── cooccurrence_network.graphml
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── data/                      # Data processing modules
│   │   ├── __init__.py
│   │   └── preprocessing.py       # Data loading and preprocessing functions
│   │
│   ├── analysis/                  # Analysis modules
│   │   ├── __init__.py
│   │   └── cooccurrence.py        # Co-occurrence analysis functions
│   │
│   └── visualization/             # Visualization modules
│       ├── __init__.py
│       ├── heatmap.py             # Heatmap generation functions
│       └── network.py             # Network graph functions
│
├── main.py                        # Main script that orchestrates the workflow
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

Output
The analysis generates several outputs:

data/processed/exploded_data.csv: Exploded view of data element - report relationships
data/processed/cooccurrence_matrix.csv: Matrix showing co-occurrence counts
output/visualizations/cooccurrence_heatmap.png: Heatmap visualization
output/visualizations/cooccurrence_network.png: Network graph visualization
output/exports/cooccurrence_network.graphml: GraphML file for further network analysis