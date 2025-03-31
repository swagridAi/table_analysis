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

class ResumableGridSearch:
    """
    Grid search with checkpoint capability to allow resuming interrupted searches.
    """
    
    def __init__(self, input_file, output_base_dir, param_ranges, max_combinations=None):
        """
        Initialize the resumable grid search.
        
        Args:
            input_file (str): Path to the input CSV file
            output_base_dir (str): Base directory for outputs
            param_ranges (dict): Dictionary with parameter names and ranges to explore
            max_combinations (int, optional): Maximum number of parameter combinations to try
        """
        self.input_file = input_file
        self.output_base_dir = output_base_dir
        self.param_ranges = param_ranges
        self.max_combinations = max_combinations
        
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
        
        # Generate all combinations of parameter values
        param_values = [self.param_ranges[name] for name in param_names]
        combinations = list(product(*param_values))
        
        # Convert to list of dictionaries
        grid = []
        for combo in combinations:
            param_dict = {name: value for name, value in zip(param_names, combo)}
            grid.append(param_dict)
        
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
        Initialize the grid search by loading data and setting up parameter combinations.
        """
        # Check if we can resume from checkpoint
        if self._load_checkpoint():
            self._log_progress(f"Resuming search from checkpoint. {len(self.results)} combinations already evaluated, {len(self.pending_combinations)} pending.")
        else:
            self._log_progress("Starting new parameter optimization.")
            # Create parameter grid
            self.pending_combinations = self._create_parameter_grid()
            self._log_progress(f"Created grid with {len(self.pending_combinations)} parameter combinations.")
        
        # Load and preprocess data (only need to do this once)
        self._log_progress("Loading and processing data...")
        df = load_data(self.input_file)
        self.df_exploded = create_exploded_dataframe(df)
        
        # Calculate co-occurrence (only need to do this once)
        self._log_progress("Calculating co-occurrence matrix...")
        co_occurrence = calculate_cooccurrence(self.df_exploded)
        self.cooccurrence_matrix, all_elements = create_cooccurrence_matrix(co_occurrence)
        
        # Get total number of reports for normalization
        self.report_count = self.df_exploded['Report'].nunique()
        
        # Create network graph (we'll reuse this for each parameter combination)
        self._log_progress("Creating network graph...")
        from networkx import Graph
        self.G = Graph()
        for (elem1, elem2), count in co_occurrence.items():
            self.G.add_edge(elem1, elem2, weight=count)
        
        self.search_initialized = True
        self._log_progress("Search initialization complete.")
    
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
            # Iterate through remaining parameter combinations
            while self.pending_combinations:
                # Get next combination
                params = self.pending_combinations.pop(0)
                
                self.iterations_completed += 1
                self._log_progress(f"Testing combination {self.iterations_completed}/{total_combinations}: {params}")
                
                try:
                    # Extract parameters
                    community_algorithm = params.get('community_algorithm', config.COMMUNITY_ALGORITHM)
                    community_resolution = params.get('community_resolution', config.COMMUNITY_RESOLUTION)
                    min_pattern_frequency = params.get('min_pattern_frequency', 2)
                    quality_weight_coverage = params.get('quality_weight_coverage', 0.5)
                    quality_weight_redundancy = params.get('quality_weight_redundancy', 0.5)
                    
                    # Detect communities with current parameters
                    communities = detect_communities(
                        self.G, 
                        algorithm=community_algorithm,
                        resolution=community_resolution
                    )
                    
                    # Convert communities to product groups
                    community_groups = []
                    for community_id in set(communities.values()):
                        group = [node for node, c_id in communities.items() if c_id == community_id]
                        community_groups.append(group)
                    
                    # Evaluate product groups with current parameters
                    metrics = evaluate_product_groups(
                        community_groups, 
                        self.cooccurrence_matrix, 
                        self.df_exploded, 
                        report_count=self.report_count,
                        min_pattern_frequency=min_pattern_frequency
                    )
                    
                    # Calculate custom quality score if weights are provided
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
                        'community_resolution': community_resolution,
                        'min_pattern_frequency': min_pattern_frequency,
                        'quality_weight_coverage': quality_weight_coverage,
                        'quality_weight_redundancy': quality_weight_redundancy,
                        'num_communities': len(community_groups),
                        'avg_community_size': np.mean([len(g) for g in community_groups]),
                        'avg_affinity_score': avg_affinity,
                        'coverage_ratio': metrics["overall_metrics"]["coverage_ratio"],
                        'weighted_coverage': coverage,
                        'redundancy_score': redundancy,
                        'quality_score': quality_score
                    }
                    
                    self.results.append(result)
                    
                    # Update optimal parameters if this is the best result so far
                    if not self.optimal_params or quality_score > self.optimal_params.get('quality_score', 0):
                        self.optimal_params = result
                        self._log_progress(f"New optimal parameters found: {community_algorithm}, resolution={community_resolution}")
                    
                except Exception as e:
                    self._log_progress(f"Error with parameters {params}: {str(e)}")
                
                # Save checkpoint after each iteration
                if self.iterations_completed % 5 == 0:  # Save every 5 iterations
                    self._save_checkpoint()
            
            # All combinations evaluated, create final output
            self._log_progress("Parameter optimization complete!")
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(self.results)
            
            # Save all results
            results_file = os.path.join(self.results_dir, "optimization_results.csv")
            results_df.to_csv(results_file, index=False)
            self._log_progress(f"Results saved to {results_file}")
            
            # Save optimal parameters
            if self.optimal_params:
                optimal_file = os.path.join(self.results_dir, "optimal_parameters.csv")
                pd.DataFrame([self.optimal_params]).to_csv(optimal_file, index=False)
                
                # Also save as JSON for easier loading
                optimal_json = os.path.join(self.results_dir, "optimal_parameters.json")
                with open(optimal_json, 'w') as f:
                    json.dump(self.optimal_params, f, indent=2)
                    
                self._log_progress(f"Optimal parameters saved to {optimal_file} and {optimal_json}")
            
            # Create visualizations
            if not results_df.empty:
                self._create_parameter_influence_plots(results_df)
            
            # Remove checkpoint file since search is complete
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                self._log_progress("Checkpoint file removed (search completed successfully).")
            
            return results_df, self.optimal_params
            
        except Exception as e:
            self._log_progress(f"Error during optimization: {str(e)}")
            self._save_checkpoint()
            self._log_progress(f"Search state saved to checkpoint. You can resume later.")
            raise
    
    def _create_parameter_influence_plots(self, results_df):
        """
        Create visualizations showing how each parameter influences the metrics.
        
        Args:
            results_df (pd.DataFrame): DataFrame with optimization results
        """
        if results_df.empty:
            return
        
        # Numeric parameters to analyze
        numeric_params = [
            'community_resolution',
            'min_pattern_frequency',
            'quality_weight_coverage',
            'quality_weight_redundancy'
        ]
        
        # Filter to parameters that exist and vary
        numeric_params = [p for p in numeric_params if p in results_df.columns and results_df[p].nunique() > 1]
        
        # Metrics to analyze
        metrics = [
            'num_communities', 
            'avg_community_size',
            'avg_affinity_score', 
            'coverage_ratio', 
            'redundancy_score', 
            'quality_score'
        ]
        
        # Filter to metrics that exist
        metrics = [m for m in metrics if m in results_df.columns]
        
        # 1. Create parameter influence plots for each numeric parameter
        for param in numeric_params:
            plt.figure(figsize=(12, 10))
            for i, metric in enumerate(metrics):
                plt.subplot(3, 2, i+1)
                
                # Plot parameter vs metric
                plt.scatter(results_df[param], results_df[metric], alpha=0.7)
                
                # Add trend line if there are enough points
                if len(results_df) > 5:
                    try:
                        z = np.polyfit(results_df[param], results_df[metric], 1)
                        p = np.poly1d(z)
                        plt.plot(sorted(results_df[param].unique()), 
                                p(sorted(results_df[param].unique())), 
                                "r--", alpha=0.7)
                    except:
                        pass  # Skip trend line if there's an error
                
                plt.title(f"{param} vs {metric}")
                plt.xlabel(param)
                plt.ylabel(metric)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, f"parameter_influence_{param}.png"), dpi=300)
            plt.close()
        
        # 2. Create algorithm comparison plot
        if 'community_algorithm' in results_df.columns and results_df['community_algorithm'].nunique() > 1:
            plt.figure(figsize=(14, 10))
            
            for i, metric in enumerate(metrics):
                plt.subplot(3, 2, i+1)
                
                # Create box plot for this metric across algorithms
                sns.boxplot(x='community_algorithm', y=metric, data=results_df)
                plt.title(f"Algorithm vs {metric}")
                plt.grid(True, alpha=0.3)
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, "algorithm_comparison.png"), dpi=300)
            plt.close()
        
        # 3. Create correlation heatmap between parameters and metrics
        # Combine parameters and metrics
        cols_to_include = [c for c in numeric_params if c in results_df.columns and results_df[c].nunique() > 1] + metrics
        if len(cols_to_include) > 1:  # Need at least 2 columns for correlation
            corr_matrix = results_df[cols_to_include].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title("Correlation between Parameters and Metrics")
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, "parameter_metric_correlation.png"), dpi=300)
            plt.close()
        
        # 4. Create quality score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['quality_score'], kde=True)
        plt.axvline(results_df['quality_score'].max(), color='r', linestyle='--', 
                    label=f'Maximum: {results_df["quality_score"].max():.4f}')
        plt.title("Distribution of Quality Scores")
        plt.xlabel("Quality Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.viz_dir, "quality_score_distribution.png"), dpi=300)
        plt.close()
        
        self._log_progress(f"Parameter influence visualizations saved to {self.viz_dir}")

def main():
    """
    Main function to run resumable parameter optimization.
    """
    parser = argparse.ArgumentParser(description='Run resumable parameter optimization')
    parser.add_argument('--input-file', '-i', default=None, 
                      help='Path to input file (default: from config.py)')
    parser.add_argument('--output-dir', '-o', default=None,
                      help='Output directory (default: timestamp-based directory)')
    parser.add_argument('--max-combinations', '-m', type=int, default=None,
                      help='Maximum number of parameter combinations to try')
    
    args = parser.parse_args()
    
    # Define parameter ranges to explore
    param_ranges = {
        'community_algorithm': ['louvain', 'label_propagation', 'greedy_modularity'],
        'community_resolution': [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        'min_pattern_frequency': [2, 3, 4],
        'quality_weight_coverage': [0.3, 0.4, 0.5, 0.6],
        'quality_weight_redundancy': [0.3, 0.4, 0.5, 0.6]
    }
    
    # Define input file
    input_file = args.input_file or config.INPUT_FILE
    
    # Define output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(config.OUTPUT_DIR, f"parameter_optimization_{timestamp}")
    else:
        output_dir = args.output_dir
    
    # Create the grid search object
    grid_search = ResumableGridSearch(
        input_file=input_file,
        output_base_dir=output_dir,
        param_ranges=param_ranges,
        max_combinations=args.max_combinations or 50  # Default limit
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