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

def load_data(filepath, clean=True, required_columns=None, verbose=True, 
              delimiter=',', header=0, index_col=None, skiprows=None, 
              na_values=None, parse_dates=False, low_memory=True, **kwargs):
    """
    Load data from CSV file with encoding detection and optional cleaning.
    
    Args:
        filepath (str): Path to the CSV file
        clean (bool): Whether to clean the data (remove rows with missing values)
        required_columns (list, optional): List of columns that must have values for cleaning
        verbose (bool): Whether to print information about data loading and cleaning
        delimiter (str): Delimiter to use for CSV file (default: ',')
        header (int, list): Row number(s) to use as column names (default: 0)
        index_col (int, str, list, None): Column(s) to use as index (default: None)
        skiprows (int, list, callable): Rows to skip (default: None)
        na_values (scalar, str, list, dict, None): Additional values to recognize as NA/NaN
        parse_dates (bool, list, dict): Identify and parse date columns (False = no parsing)
        low_memory (bool): Internally process the file in chunks (helps for large files)
        **kwargs: Additional keyword arguments to pass to pandas.read_csv()
        
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
            # Pass the explicit parameters and any additional kwargs to read_csv
            df = pd.read_csv(
                filepath, 
                encoding=encoding,
                delimiter=delimiter,
                header=header,
                index_col=index_col,
                skiprows=skiprows,
                na_values=na_values,
                parse_dates=parse_dates,
                low_memory=low_memory,
                **kwargs
            )
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
        if df.shape[0] > 0:
            memory_usage = df.memory_usage(deep=True).sum()
            print(f"DataFrame memory usage: {memory_usage / 1048576:.2f} MB")
        
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
    
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Create a unique identifier for each data element
    df["data_element"] = df["Data Element Table"] + "." + df["Data Element Column"]
    
    # Filter out rows with NaN values in Enterprise Report Catalog
    df_filtered = df.dropna(subset=["Enterprise Report Catalog"])
    
    # Convert Enterprise Report Catalog to string if it's not already
    df_filtered["Enterprise Report Catalog"] = df_filtered["Enterprise Report Catalog"].astype(str)
    
    # Split the reports by newline and create lists
    df_filtered["Reports_List"] = df_filtered["Enterprise Report Catalog"].str.strip().str.split('\n')
    
    # Explode the dataframe by the reports list
    exploded_df = df_filtered.explode("Reports_List")
    
    # Clean up reports (remove empty strings)
    exploded_df = exploded_df[exploded_df["Reports_List"].str.strip() != ""]
    exploded_df["Reports_List"] = exploded_df["Reports_List"].str.strip()
    
    # Check if we have any exploded rows
    if len(exploded_df) == 0:
        raise ValueError("No valid report-element pairs found after processing")
    
    # Create the final dataframe with the desired columns
    result_df = pd.DataFrame({
        "Report": exploded_df["Reports_List"],
        "Table": exploded_df["Data Element Table"],
        "Column": exploded_df["Data Element Column"],
        "data_element": exploded_df["data_element"]
    })
    
    # Add the CriticalFlag column, defaulting to empty string if not present
    if "Critical Data Element" in df.columns:
        result_df["CriticalFlag"] = exploded_df["Critical Data Element"]
    else:
        result_df["CriticalFlag"] = ""
    
    return result_df