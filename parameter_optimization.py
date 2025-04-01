#!/usr/bin/env python3
"""
Resumable parameter optimization for Data Co-occurrence Analysis.

This script performs grid search over model parameters with checkpoint capability,
allowing the search to be resumed if interrupted. It saves progress after each
parameter combination evaluation and can pick up where it left off when restarted.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import signal
import pickle
from itertools import product
from datetime import datetime
import argparse
import sys

from src.data.preprocessing import load_data, create_exploded_dataframe
from src.analysis.cooccurrence import calculate_cooccurrence, create_cooccurrence_matrix
from src.analysis.clustering import detect_communities, analyze_communities
from src.analysis.metrics import evaluate_product_groups
import config

def debug_type_info(value1, value2, context=""):
    """Print debug information about types when comparison fails."""
    print(f"\n*** TYPE COMPARISON DEBUG ({context}) ***")
    print(f"Value 1: '{value1}' of type {type(value1)}")
    print(f"Value 2: '{value2}' of type {type(value2)}")
    print("*** END DEBUG ***\n")

class ResumableGridSearch:
    """
    Grid search with checkpoint capability to allow resuming interrupted searches.
    """
    
    def __init__(self, input_file, output_base_dir, param_ranges, max_combinations=None, validate=True, force=False):
        """
        Initialize the resumable grid search.
        
        Args:
            input_file (str): Path to the input CSV file
            output_base_dir (str): Base directory for outputs
            param_ranges (dict): Dictionary with parameter names and ranges to explore
            max_combinations (int, optional): Maximum number of parameter combinations to try
            validate (bool): Whether to run validation before starting
            force (bool): Whether to force execution even if validation fails
        """
        self.input_file = input_file
        self.output_base_dir = output_base_dir
        self.param_ranges = param_ranges
        self.max_combinations = max_combinations
        
        # Run validation if requested
        if validate:
            from optimization_validation import run_optimization_validation
            validation_passed = run_optimization_validation(
                input_file, output_base_dir, param_ranges
            )
            if not validation_passed and not force:
                raise ValueError("Validation failed. Use force=True to run anyway.")
        
        # Create necessary directories
        self.results_dir, self.viz_dir, self.logs_dir = self._ensure_directories_exist()
        
        # Timestamp for logging
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.logs_dir, f"optimization_log_{self.timestamp}.txt")
        
        # Checkpoint file path
        self.checkpoint_file = os.path.join(self.output_base_dir, "search_checkpoint.pkl")
        
        # Progress tracking
        self.results = []
        self.pending_combinations = []
        self.optimal_params = None
        self.iterations_completed = 0
        self.search_initialized = False
        
        # Data processing objects (will be initialized during run)
        self.df_exploded = None
        self.cooccurrence_matrix = None
        self.report_count = None
        
        # Setup signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _ensure_directories_exist(self):
        """Create output directories if they don't exist."""
        for directory in [
            os.path.join(self.output_base_dir, "results"),
            os.path.join(self.output_base_dir, "visualizations"),
            os.path.join(self.output_base_dir, "logs")
        ]:
            os.makedirs(directory, exist_ok=True)
        
        return (os.path.join(self.output_base_dir, "results"), 
                os.path.join(self.output_base_dir, "visualizations"), 
                os.path.join(self.output_base_dir, "logs"))
    
    def _log_progress(self, message):
        """Log progress to a file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)
    
    def _signal_handler(self, sig, frame):
        """Handle interruption signals by saving checkpoint before exiting."""
        self._log_progress("\nSearch interrupted. Saving checkpoint...")
        self._save_checkpoint()
        self._log_progress(f"Checkpoint saved to {self.checkpoint_file}")
        self._log_progress("You can resume the search by running the script again with the same output directory.")
        sys.exit(0)
    
    def _create_parameter_grid(self):
        """
        Create a grid of all parameter combinations to explore.
        
        Returns:
            list: List of dictionaries, each representing a parameter combination
        """
        # Get all parameter names
        param_names = list(self.param_ranges.keys())
        
        # Sanitize parameter ranges
        sanitized_ranges = {}
        for name, values in self.param_ranges.items():
            if name in ['community_resolution', 'min_pattern_frequency', 
                    'quality_weight_coverage', 'quality_weight_redundancy']:
                # Convert numeric parameters to float
                sanitized_values = [float(v) if isinstance(v, (int, str)) and str(v).replace('.', '', 1).isdigit() else v 
                                for v in values]
                sanitized_ranges[name] = sanitized_values
                
                # Log any changes
                if sanitized_values != values:
                    print(f"Sanitized parameter {name}: {values} -> {sanitized_values}")
            else:
                sanitized_ranges[name] = values
        
        # Use sanitized ranges
        self.param_ranges = sanitized_ranges
        
        # Generate all combinations of parameter values
        param_values = [self.param_ranges[name] for name in param_names]
        combinations = list(product(*param_values))
        
        # Convert to list of dictionaries
        grid = []
        for combo in combinations:
            try:
                param_dict = {name: value for name, value in zip(param_names, combo)}
                grid.append(param_dict)
            except Exception as e:
                # Debug the values that caused the issue
                print(f"Error creating parameter combination: {str(e)}")
                for i, (name, value) in enumerate(zip(param_names, combo)):
                    print(f"Parameter {name}: '{value}' of type {type(value)}")
                raise
        
        # Shuffle to ensure diverse sampling in case of interruption
        np.random.shuffle(grid)
        
        # Limit combinations if requested
        if self.max_combinations and len(grid) > self.max_combinations:
            grid = grid[:self.max_combinations]
        
        return grid
    
    def _save_checkpoint(self):
        """Save current state to checkpoint file."""
        checkpoint_data = {
            'results': self.results,
            'pending_combinations': self.pending_combinations,
            'optimal_params': self.optimal_params,
            'iterations_completed': self.iterations_completed,
            'timestamp': self.timestamp,
            'param_ranges': self.param_ranges,
            'max_combinations': self.max_combinations
        }
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Also save current results to CSV for easy inspection
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_file = os.path.join(self.results_dir, "optimization_results_partial.csv")
            results_df.to_csv(results_file, index=False)
    
    def _load_checkpoint(self):
        """
        Load state from checkpoint file.
        
        Returns:
            bool: True if checkpoint was loaded, False otherwise
        """
        if not os.path.exists(self.checkpoint_file):
            return False
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.results = checkpoint_data.get('results', [])
            self.pending_combinations = checkpoint_data.get('pending_combinations', [])
            self.optimal_params = checkpoint_data.get('optimal_params', None)
            self.iterations_completed = checkpoint_data.get('iterations_completed', 0)
            self.timestamp = checkpoint_data.get('timestamp', self.timestamp)
            
            # Update log file path with loaded timestamp
            self.log_file = os.path.join(self.logs_dir, f"optimization_log_{self.timestamp}.txt")
            
            return True
        except Exception as e:
            self._log_progress(f"Error loading checkpoint: {str(e)}")
            return False
    
    def initialize_search(self):
        """
        Initialize the grid search by loading data and setting up parameter combinations
        with memory-efficient processing.
        """
        import gc
        import psutil
        import os
        from scipy import sparse
        from networkx import Graph
        
        # Setup memory monitoring
        def log_memory_usage(label=""):
            """Log current memory usage."""
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self._log_progress(f"Memory usage {label}: {memory_mb:.2f} MB")
        
        # Check if we can resume from checkpoint
        if self._load_checkpoint():
            self._log_progress(f"Resuming search from checkpoint. {len(self.results)} combinations already evaluated, {len(self.pending_combinations)} pending.")
        else:
            self._log_progress("Starting new parameter optimization.")
            # Create parameter grid
            self.pending_combinations = self._create_parameter_grid()
            self._log_progress(f"Created grid with {len(self.pending_combinations)} parameter combinations.")
        
        log_memory_usage("Before data loading")
        
        # Determine required columns to reduce memory footprint
        required_columns = ["Data Element Table", "Data Element Column", "Enterprise Report Catalog"]
        
        # Process data in chunks instead of loading all at once
        self._log_progress("Loading and processing data in chunks...")
        chunk_size = 10000  # Adjust based on available memory and dataset size
        
        # Initialize co-occurrence dictionary with frequency tracking
        co_occurrence = {}
        report_set = set()  # Track unique reports
        
        # Track how many chunks we've processed
        chunk_count = 0
        
        # Process file in chunks to reduce memory usage
        for chunk in pd.read_csv(self.input_file, chunksize=chunk_size, usecols=required_columns):
            chunk_count += 1
            self._log_progress(f"Processing chunk {chunk_count}...")
            
            # Process this chunk
            chunk_exploded = create_exploded_dataframe(chunk)
            
            # Track unique reports
            report_set.update(chunk_exploded['Report'].unique())
            
            # Calculate co-occurrence for this chunk
            chunk_cooc = calculate_cooccurrence(chunk_exploded)
            
            # Merge with existing co-occurrence data
            for key, count in chunk_cooc.items():
                co_occurrence[key] = co_occurrence.get(key, 0) + count
            
            # Clear chunk data to free memory
            del chunk
            del chunk_exploded
            del chunk_cooc
            gc.collect()
            
            log_memory_usage(f"After processing chunk {chunk_count}")
        
        # Store the total report count
        self.report_count = len(report_set)
        self._log_progress(f"Found {self.report_count} unique reports.")
        
        # Use disk-based storage for very large datasets
        use_disk_storage = len(co_occurrence) > 1000000  # Threshold for disk storage
        
        if use_disk_storage:
            # Store co-occurrence data on disk
            from src.utils.disk_storage import DiskBasedCooccurrenceMatrix
            matrix_file = os.path.join(self.output_base_dir, "cooccurrence_matrix.h5")
            self._log_progress(f"Using disk-based storage for co-occurrence matrix: {matrix_file}")
            self.matrix_storage = DiskBasedCooccurrenceMatrix(matrix_file)
            self.matrix_storage.store_matrix(co_occurrence)
            self.all_elements = self.matrix_storage.all_elements
            self.cooccurrence_matrix = None  # Not using in-memory matrix
        else:
            # Create sparse co-occurrence matrix
            self._log_progress("Creating sparse co-occurrence matrix...")
            
            # Get all unique elements
            all_elements = sorted(set(elem for pair in co_occurrence.keys() for elem in pair))
            self.all_elements = all_elements
            elem_to_idx = {elem: i for i, elem in enumerate(all_elements)}
            
            # Create sparse matrix in COO format (memory efficient for construction)
            rows = []
            cols = []
            data = []
            
            for (elem1, elem2), count in co_occurrence.items():
                i = elem_to_idx[elem1]
                j = elem_to_idx[elem2]
                rows.append(i)
                cols.append(j)
                data.append(count)
                # Also add symmetric entry
                rows.append(j)
                cols.append(i)
                data.append(count)
            
            # Create sparse matrix
            n = len(all_elements)
            matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
            
            # Convert to CSR format for efficient operations
            self.cooccurrence_matrix = matrix.tocsr()
            self.matrix_storage = None  # Not using disk storage
        
        log_memory_usage("After creating co-occurrence matrix")
        
        # Clear raw co-occurrence dictionary to save memory
        del co_occurrence
        gc.collect()
        
        # Create memory-efficient network graph (filtering low-frequency edges)
        self._log_progress("Creating network graph...")
        min_edge_weight = 2  # Only include edges with weight >= 2
        self.G = Graph()
        
        # Add nodes and edges efficiently
        if use_disk_storage:
            # Add nodes first
            for elem in self.all_elements:
                self.G.add_node(elem)
            
            # Add edges from disk storage (using memory-efficient approach)
            rows = self.matrix_storage.file['matrix/rows'][()]
            cols = self.matrix_storage.file['matrix/cols'][()]
            data = self.matrix_storage.file['matrix/data'][()]
            
            # Process in chunks to avoid memory issues
            chunk_size = 100000
            for i in range(0, len(rows), chunk_size):
                end = min(i + chunk_size, len(rows))
                for idx in range(i, end):
                    if data[idx] >= min_edge_weight:
                        row_idx = rows[idx]
                        col_idx = cols[idx]
                        # Only add each edge once (avoid duplicates from symmetric matrix)
                        if row_idx < col_idx:
                            elem1 = self.all_elements[row_idx]
                            elem2 = self.all_elements[col_idx]
                            self.G.add_edge(elem1, elem2, weight=data[idx])
        else:
            # Add nodes and edges from sparse matrix
            for elem in self.all_elements:
                self.G.add_node(elem)
            
            # Get edges from sparse matrix
            cx = self.cooccurrence_matrix.tocoo()
            for i, j, v in zip(cx.row, cx.col, cx.data):
                if i < j and v >= min_edge_weight:  # Add each edge only once
                    elem1 = self.all_elements[i]
                    elem2 = self.all_elements[j]
                    self.G.add_edge(elem1, elem2, weight=v)
        
        # Keep a small exploded dataframe for pattern evaluation
        # Load a small sample for metrics calculation
        sample_size = min(50000, self.report_count)  # Limit size for memory efficiency
        self._log_progress(f"Loading sample of {sample_size} reports for metrics calculation...")
        
        df_sample = pd.read_csv(self.input_file, nrows=sample_size, usecols=required_columns)
        self.df_exploded = create_exploded_dataframe(df_sample)
        
        log_memory_usage("After graph creation")
        
        # Mark initialization as complete
        self.search_initialized = True
        self._log_progress("Search initialization complete.")
        self._log_progress(f"Graph contains {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")
        
        # Final garbage collection
        gc.collect()
        log_memory_usage("Final")
    
    def run(self):
        """
        Run the parameter optimization with checkpoint capability.
        
        Returns:
            tuple: (pd.DataFrame with all results, dict with optimal parameters)
        """
        if not self.search_initialized:
            self.initialize_search()
        
        total_combinations = len(self.results) + len(self.pending_combinations)
        self._log_progress(f"Running parameter optimization with {total_combinations} total combinations.")
        self._log_progress(f"{len(self.results)} already evaluated, {len(self.pending_combinations)} remaining.")
        
        try:
            # Process all pending parameter combinations
            self._process_parameter_combinations(total_combinations)
            
            # Generate final outputs
            results_df = self._generate_final_outputs()
            
            return results_df, self.optimal_params
                
        except Exception as e:
            self._log_progress(f"Error during optimization: {str(e)}")
            self._save_checkpoint()
            self._log_progress(f"Search state saved to checkpoint. You can resume later.")
            raise

    def _process_parameter_combinations(self, total_combinations):
        """
        Process all pending parameter combinations.
        
        Args:
            total_combinations (int): Total number of combinations to process
        """
        # Iterate through remaining parameter combinations
        while self.pending_combinations:
            # Get next combination
            params = self.pending_combinations.pop(0)
            
            self.iterations_completed += 1
            self._log_progress(f"Testing combination {self.iterations_completed}/{total_combinations}: {params}")
            
            try:
                # Evaluate the current parameter combination
                result = self._evaluate_parameter_combination(params)
                
                if result:
                    self.results.append(result)
                    # Update optimal parameters
                    self._update_optimal_parameters(result)
            
            except Exception as e:
                self._log_progress(f"Error with parameters {params}: {str(e)}")
            
            # Save checkpoint periodically
            if self.iterations_completed % 5 == 0:  # Save every 5 iterations
                self._save_checkpoint()

    def _evaluate_parameter_combination(self, params):
        """
        Evaluate a single parameter combination.
        
        Args:
            params (dict): Parameter combination to evaluate
            
        Returns:
            dict: Results for this parameter combination, or None if evaluation failed
        """
        try:
            # Extract parameters
            community_algorithm = params.get('community_algorithm', config.COMMUNITY_ALGORITHM)
            community_resolution = params.get('community_resolution', config.COMMUNITY_RESOLUTION)
            min_pattern_frequency = params.get('min_pattern_frequency', 2)
            quality_weight_coverage = params.get('quality_weight_coverage', 0.5)
            quality_weight_redundancy = params.get('quality_weight_redundancy', 0.5)
            
            # Ensure parameter types
            community_resolution = float(community_resolution)
            min_pattern_frequency = float(min_pattern_frequency)
            quality_weight_coverage = float(quality_weight_coverage)
            quality_weight_redundancy = float(quality_weight_redundancy)
            
        except Exception as e:
            self._log_progress(f"Error extracting parameters: {str(e)}")
            # Print all parameters for debugging
            for k, v in params.items():
                self._log_progress(f"  {k}: '{v}' of type {type(v)}")
            return None
        
        # Detect communities with current parameters
        communities = detect_communities(
            self.G, 
            algorithm=community_algorithm,
            resolution=community_resolution
        )
        
        # Convert communities to product groups
        community_groups = self._convert_communities_to_groups(communities)
        
        # Evaluate product groups with current parameters
        metrics = evaluate_product_groups(
            community_groups, 
            self.cooccurrence_matrix, 
            self.df_exploded, 
            report_count=self.report_count,
            min_pattern_frequency=min_pattern_frequency
        )
        
        # Calculate quality metrics
        result = self._calculate_quality_metrics(
            metrics, community_groups, 
            community_algorithm, community_resolution, 
            min_pattern_frequency, quality_weight_coverage, quality_weight_redundancy
        )
        
        return result

    def _convert_communities_to_groups(self, communities):
        """
        Convert communities dictionary to list of community groups.
        
        Args:
            communities (dict): Dictionary mapping nodes to community IDs
            
        Returns:
            list: List of community groups, where each group is a list of nodes
        """
        community_groups = []
        for community_id in set(communities.values()):
            group = [node for node, c_id in communities.items() if c_id == community_id]
            community_groups.append(group)
        
        return community_groups

    def _calculate_quality_metrics(self, metrics, community_groups, 
                                community_algorithm, community_resolution,
                                min_pattern_frequency, quality_weight_coverage, quality_weight_redundancy):
        """
        Calculate quality metrics for the current parameter combination.
        
        Args:
            metrics (dict): Metrics from evaluate_product_groups
            community_groups (list): List of community groups
            community_algorithm (str): Community detection algorithm used
            community_resolution (float): Resolution parameter
            min_pattern_frequency (float): Minimum pattern frequency
            quality_weight_coverage (float): Weight for coverage in quality score
            quality_weight_redundancy (float): Weight for redundancy in quality score
            
        Returns:
            dict: Result dictionary with all metrics
        """
        # Calculate metrics
        avg_affinity = np.mean([g["affinity_score"] for g in metrics["group_metrics"]])
        coverage = metrics["overall_metrics"]["weighted_coverage"]
        redundancy = metrics["overall_metrics"]["redundancy_score"]
        
        # Weighted quality score
        quality_score = (avg_affinity * (1-quality_weight_coverage-quality_weight_redundancy) + 
                        coverage * quality_weight_coverage) * (1 - redundancy * quality_weight_redundancy)
        
        # Store results
        result = {
            'combination_id': self.iterations_completed,
            'community_algorithm': community_algorithm,
            'community_resolution': float(community_resolution),
            'min_pattern_frequency': float(min_pattern_frequency),
            'quality_weight_coverage': float(quality_weight_coverage),
            'quality_weight_redundancy': float(quality_weight_redundancy),
            'num_communities': len(community_groups),
            'avg_community_size': np.mean([len(g) for g in community_groups]),
            'avg_affinity_score': float(avg_affinity),
            'coverage_ratio': float(metrics["overall_metrics"]["coverage_ratio"]),
            'weighted_coverage': float(coverage),
            'redundancy_score': float(redundancy),
            'quality_score': float(quality_score)
        }
        
        return result

    def _update_optimal_parameters(self, result):
        """
        Update optimal parameters if the current result is better.
        
        Args:
            result (dict): Current result to evaluate
        """
        current_score = float(result['quality_score'])
        previous_best = float(self.optimal_params['quality_score']) if self.optimal_params else 0

        if not self.optimal_params or current_score > previous_best:
            self.optimal_params = result
            self._log_progress(f"New optimal parameters found: {result['community_algorithm']}, resolution={result['community_resolution']}")

    def _generate_final_outputs(self):
        """
        Generate and save final outputs after optimization is complete.
        
        Returns:
            pd.DataFrame: DataFrame with all results
        """
        self._log_progress("Parameter optimization complete!")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save all results
        if not results_df.empty:
            results_file = os.path.join(self.results_dir, "optimization_results.csv")
            results_df.to_csv(results_file, index=False)
            self._log_progress(f"Results saved to {results_file}")
        
        # Save optimal parameters
        if self.optimal_params:
            self._save_optimal_parameters()
        
        # Create visualizations if we have results
        if not results_df.empty:
            self._create_visualizations(results_df)
        
        # Remove checkpoint file since search is complete
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            self._log_progress("Checkpoint file removed (search completed successfully).")
        
        return results_df

    def _save_optimal_parameters(self):
        """Save optimal parameters to CSV and JSON files."""
        optimal_file = os.path.join(self.results_dir, "optimal_parameters.csv")
        pd.DataFrame([self.optimal_params]).to_csv(optimal_file, index=False)
        
        # Also save as JSON for easier loading
        optimal_json = os.path.join(self.results_dir, "optimal_parameters.json")
        with open(optimal_json, 'w') as f:
            json.dump(self.optimal_params, f, indent=2)
            
        self._log_progress(f"Optimal parameters saved to {optimal_file} and {optimal_json}")

    def _create_visualizations(self, results_df):
        """
        Create visualizations from optimization results.
        
        Args:
            results_df (pd.DataFrame): DataFrame with optimization results
        """
        try:
            from src.visualization.parameter_plots import create_parameter_influence_plots
            create_parameter_influence_plots(
                results_df, 
                self.viz_dir,
                logger=self._log_progress
            )
            self._log_progress("Visualizations created successfully.")
        except Exception as e:
            self._log_progress(f"Error creating visualizations: {str(e)}")
    

def ensure_parameter_types(param_ranges):
    """Ensure all parameters have consistent types."""
    numeric_param_names = [
        'community_resolution', 
        'min_pattern_frequency',
        'quality_weight_coverage', 
        'quality_weight_redundancy'
    ]
    
    for param_name in numeric_param_names:
        if param_name in param_ranges:
            # Convert all values to float
            param_ranges[param_name] = [float(value) for value in param_ranges[param_name]]
    
    return param_ranges

def inspect_input_data(input_file):
    """Inspect the input data for potential issues."""
    print(f"\n=== INSPECTING DATA FROM {input_file} ===")
    try:
        # Load the data
        df = load_data(input_file)
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        print("\nMissing value counts:")
        print(missing_counts[missing_counts > 0] if any(missing_counts > 0) else "No missing values")
        
        # Check data types
        print("\nColumn data types:")
        print(df.dtypes)
        
        # Check for mixed types in important columns
        for col in ["Data Element Table", "Data Element Column", "Enterprise Report Catalog"]:
            if col in df.columns:
                types_in_column = df[col].apply(type).value_counts()
                print(f"\nTypes in {col}:")
                print(types_in_column)
        
        # Sample values from key columns
        print("\nSample values:")
        for col in df.columns:
            unique_values = df[col].dropna().unique()[:3]  # First 3 unique values
            print(f"{col}: {unique_values}")
            
        return df
    except Exception as e:
        print(f"Error inspecting data: {str(e)}")
        return None

def main():
    """
    Main function to run parameter optimization.
    """
    parser = argparse.ArgumentParser(description='Run resumable parameter optimization')
    parser.add_argument('--input-file', '-i', default=None, 
                      help='Path to input file (default: from config.py)')
    parser.add_argument('--output-dir', '-o', default=None,
                      help='Output directory (default: timestamp-based directory)')
    parser.add_argument('--max-combinations', '-m', type=int, default=None,
                      help='Maximum number of parameter combinations to try')
    parser.add_argument('--no-validate', '-n', action='store_true',
                      help='Skip validation checks')
    parser.add_argument('--force', '-f', action='store_true',
                      help='Force execution even if validation fails')
    
    args = parser.parse_args()
    
    # Define parameter ranges to explore
    param_ranges = {
        'community_algorithm': ['louvain', 'label_propagation', 'greedy_modularity'],
        'community_resolution': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        'min_pattern_frequency': [2, 3, 4],
        'quality_weight_coverage': [0.3, 0.4, 0.5, 0.6],
        'quality_weight_redundancy': [0.3, 0.4, 0.5, 0.6]
    }
    param_ranges = ensure_parameter_types(param_ranges)
    # Define input file
    input_file = args.input_file or config.INPUT_FILE
    
    # Define output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(config.OUTPUT_DIR, f"parameter_optimization_{timestamp}")
    else:
        output_dir = args.output_dir
    
    # Ensure input data is clean 
    input_df = inspect_input_data(input_file)
    # Create the grid search object
    grid_search = ResumableGridSearch(
        input_file=input_file,
        output_base_dir=output_dir,
        param_ranges=param_ranges,
        max_combinations=args.max_combinations or 50,  # Default limit
        validate=not args.no_validate,
        force=args.force
    )
    
    # Run the search
    results_df, optimal_params = grid_search.run()
    
    if optimal_params:
        print("\nOptimal Parameters:")
        for param, value in optimal_params.items():
            if param not in ['combination_id'] and not param.startswith('_'):
                print(f"  {param}: {value}")
    
    print(f"\nAll results and visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()