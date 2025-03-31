"""
Functions for loading and preprocessing data for co-occurrence analysis.
"""

import pandas as pd

def load_data(filepath):
    """
    Load data from CSV file with encoding detection.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    # Try different encodings
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
    
    for encoding in encodings_to_try:
        try:
            print(f"Trying to load file with {encoding} encoding...")
            return pd.read_csv(filepath, encoding=encoding)
        except UnicodeDecodeError:
            print(f"Failed with {encoding} encoding, trying next...")
            continue
        except Exception as e:
            # For other errors, just raise them
            raise e
    
    # If we get here, all encodings failed
    raise ValueError(f"Could not decode the file {filepath} with any of the attempted encodings: {encodings_to_try}")

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
        # Handle NaN or non-string values in the reports column
        reports_value = row["Enterprise Report Catalog"]
        
        # Skip if NaN
        if pd.isna(reports_value):
            continue
            
        # Convert to string if it's not already
        if not isinstance(reports_value, str):
            reports_value = str(reports_value)
        
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