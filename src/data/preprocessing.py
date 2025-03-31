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
    # Handle case sensitivity in column names
    expected_columns = ['Data Element Table', 'Data Element Column', 'Enterprise Report Catalog', 'Critical Data Element']
    column_mapping = {}
    
    # Create a case-insensitive mapping of columns
    for expected_col in expected_columns:
        for actual_col in df.columns:
            if expected_col.lower() == actual_col.lower():
                column_mapping[expected_col] = actual_col
                break
    
    # Check if all required columns were found
    missing_columns = [col for col in expected_columns if col not in column_mapping]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {list(df.columns)}")
    
    # Create a unique identifier for each data element
    df["data_element"] = df[column_mapping["Data Element Table"]] + "." + df[column_mapping["Data Element Column"]]
    
    # Explode the dataframe by report
    exploded_rows = []
    for idx, row in df.iterrows():
        # Handle NaN or non-string values in the reports column
        reports_value = row[column_mapping["Enterprise Report Catalog"]]
        
        # Skip if NaN
        if pd.isna(reports_value):
            print(f"Warning: Row {idx} has NaN in the '{column_mapping['Enterprise Report Catalog']}' column")
            continue
            
        # Convert to string if it's not already
        if not isinstance(reports_value, str):
            reports_value = str(reports_value)
            print(f"Warning: Row {idx} has a non-string value in '{column_mapping['Enterprise Report Catalog']}' column: {reports_value}")
        
        # Split by newline
        reports = reports_value.split("\n")
        
        for report in reports:
            exploded_rows.append({
                "Report": report.strip(),
                "CriticalFlag": row[column_mapping["Critical Data Element"]],
                "Table": row[column_mapping["Data Element Table"]],
                "Column": row[column_mapping["Data Element Column"]],
                "data_element": row["data_element"]
            })
    
    return pd.DataFrame(exploded_rows)