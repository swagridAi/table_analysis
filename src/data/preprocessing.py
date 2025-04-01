def clean_data(df, required_columns=None, verbose=True):
    """
    Clean input data by removing rows with missing values and ensuring consistent data types.
    
    Args:
        df (pandas.DataFrame): Input dataframe to clean
        required_columns (list, optional): List of columns that must have values.
            If None, defaults to ["Data Element Table", "Data Element Column", "Enterprise Report Catalog"]
        verbose (bool): Whether to print information about cleaning actions
        
    Returns:
        pandas.DataFrame: Cleaned dataframe
    """
    import pandas as pd
    
    # Store original row count
    original_count = len(df)
    
    # Set default required columns if not specified
    if required_columns is None:
        required_columns = ["Data Element Table", "Data Element Column", "Enterprise Report Catalog"]
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing from input data: {missing_columns}")
    
    # Remove rows with NaN values in required columns
    df_clean = df.dropna(subset=required_columns)
    
    # Check for and convert data types if needed
    if "Critical Data Element" in df_clean.columns:
        # Ensure Critical Data Element is string type
        df_clean["Critical Data Element"] = df_clean["Critical Data Element"].astype(str)
    
    # Report on cleaning actions
    if verbose:
        rows_removed = original_count - len(df_clean)
        if rows_removed > 0:
            print(f"Removed {rows_removed} rows ({rows_removed/original_count:.1%}) with missing values")
            
        # Report on data types
        for col in required_columns:
            type_counts = df_clean[col].apply(type).value_counts()
            if len(type_counts) > 1:
                print(f"Warning: Column '{col}' has mixed types:")
                for type_name, count in type_counts.items():
                    print(f"  {type_name.__name__}: {count} values")
    
    return df_clean

def load_data(filepath, clean=True, required_columns=None, verbose=True):
    """
    Load data from CSV file with encoding detection and optional cleaning.
    
    Args:
        filepath (str): Path to the CSV file
        clean (bool): Whether to clean the data (remove rows with missing values)
        required_columns (list, optional): List of columns that must have values for cleaning
        verbose (bool): Whether to print information about data loading and cleaning
        
    Returns:
        pandas.DataFrame: Loaded and optionally cleaned dataframe
    """
    import pandas as pd
    
    # Try different encodings
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
    
    df = None
    for encoding in encodings_to_try:
        try:
            if verbose:
                print(f"Trying to load file with {encoding} encoding...")
            df = pd.read_csv(filepath, encoding=encoding)
            break
        except UnicodeDecodeError:
            if verbose:
                print(f"Failed with {encoding} encoding, trying next...")
            continue
        except Exception as e:
            # For other errors, just raise them
            raise e
    
    # If we get here and df is None, all encodings failed
    if df is None:
        raise ValueError(f"Could not decode the file {filepath} with any of the attempted encodings: {encodings_to_try}")
    
    # Apply data cleaning if requested
    if clean:
        df = clean_data(df, required_columns=required_columns, verbose=verbose)
    
    # Print data summary
    if verbose:
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        
    return df

# Update create_exploded_dataframe to handle potential issues better
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
    import pandas as pd
    
    # Additional validation
    required_columns = ["Data Element Table", "Data Element Column", "Enterprise Report Catalog"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")
    
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
        
        # Split by newline to get individual reports
        reports = reports_value.strip().split("\n")
        
        # Skip empty reports
        reports = [report.strip() for report in reports if report.strip()]
        
        for report in reports:
            exploded_rows.append({
                "Report": report,
                "CriticalFlag": row.get("Critical Data Element", ""),
                "Table": row["Data Element Table"],
                "Column": row["Data Element Column"],
                "data_element": row["data_element"]
            })
    
    # Check if we have any exploded rows
    if not exploded_rows:
        raise ValueError("No valid report-element pairs found after processing")
    
    return pd.DataFrame(exploded_rows)