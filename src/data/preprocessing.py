"""
Functions for loading and preprocessing data for co-occurrence analysis.
"""

import pandas as pd

def load_data(filepath):
    """
    Load data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    return pd.read_csv(filepath)

def create_exploded_dataframe(df):
    """
    Create an exploded dataframe where each row represents a single report-data element pair.
    
    Args:
        df (pandas.DataFrame): Input dataframe with columns 'Data Element Table', 
                               'Data Element Column', and 'Enterprise Report Catalog'
        
    Returns:
        pandas.DataFrame: Exploded dataframe with columns 'Report', 'CriticalFlag', 
                          'Table', 'Column', and 'data_element'
    """
    # Create a unique identifier for each data element
    df["data_element"] = df["Data Element Table"] + "." + df["Data Element Column"]
    
    # Explode the dataframe by report
    exploded_rows = []
    for idx, row in df.iterrows():
        reports = row["Enterprise Report Catalog"].split("\n")
        for report in reports:
            exploded_rows.append({
                "Report": report.strip(),
                "CriticalFlag": row["Critical Data Element"],
                "Table": row["Data Element Table"],
                "Column": row["Data Element Column"],
                "data_element": row["data_element"]
            })
    
    return pd.DataFrame(exploded_rows)