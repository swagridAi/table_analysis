"""
Command-line interface for the checkpoint management system.

This module provides the command-line interface for interacting with
checkpoint management functionality.
"""

import argparse
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd

from .manager import CheckpointManager
from .visualization import visualize_parameter_space, visualize_explored_vs_pending
from .suggestion_strategies import suggest_parameter_combinations
from .validation import validate_and_print
from .param_utils import get_numeric_params


def display_status(status: Dict[str, Any]) -> None:
    """
    Display search status in a formatted way.
    
    Args:
        status: Dictionary with status information
    """
    print("\nSearch Status:")
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


def handle_status_command(manager: CheckpointManager) -> None:
    """
    Handle the status command - display optimization status.
    
    Args:
        manager: Checkpoint manager instance
    """
    status = manager.get_status()
    display_status(status)


def handle_visualization_command(manager: CheckpointManager, output_dir: str) -> None:
    """
    Handle the visualization command - create parameter space visualizations.
    
    Args:
        manager: Checkpoint manager instance
        output_dir: Output directory for visualizations
    """
    print("\nCreating parameter space visualizations...")
    
    # Get data for visualization
    results_df = manager.get_results_dataframe()
    param_ranges = manager.get_parameter_ranges()
    
    # Create visualizations
    visualize_parameter_space(results_df, param_ranges, output_dir)
    
    # Create explored vs pending visualization if there are pending combinations
    pending_df = manager.get_pending_dataframe()
    if not pending_df.empty:
        visualize_explored_vs_pending(results_df, pending_df, output_dir)
    
    print(f"Visualizations saved to {output_dir}")


def handle_export_command(manager: CheckpointManager, export_dir: str) -> None:
    """
    Handle the export command - export results and metadata.
    
    Args:
        manager: Checkpoint manager instance
        export_dir: Directory to save exports
    """
    print(f"\nExporting results to {export_dir}...")
    success = manager.export_results(export_dir)
    
    if success:
        print(f"Export complete. Files saved to {export_dir}")
    else:
        print("Export failed. See error messages for details.")


def handle_suggest_command(
    manager: CheckpointManager, 
    num_suggestions: int, 
    output_dir: str
) -> None:
    """
    Handle the suggest command - generate parameter suggestions.
    
    Args:
        manager: Checkpoint manager instance
        num_suggestions: Number of suggestions to generate
        output_dir: Output directory for saving suggestions
    """
    print(f"\nGenerating {num_suggestions} suggested parameter combinations...")
    
    # Get data for suggestion generation
    results_df = manager.get_results_dataframe()
    param_ranges = manager.get_parameter_ranges()
    numeric_params = get_numeric_params(results_df)
    
    # Generate suggestions
    suggestions = suggest_parameter_combinations(
        results_df, 
        param_ranges, 
        numeric_params, 
        num_suggestions
    )
    
    # Create DataFrame and save suggestions
    if suggestions:
        suggestions_df = pd.DataFrame(suggestions)
        
        # Save suggestions to file
        os.makedirs(output_dir, exist_ok=True)
        suggestions_file = os.path.join(output_dir, "suggested_combinations.csv")
        suggestions_df.to_csv(suggestions_file, index=False)
        print(f"Suggestions saved to {suggestions_file}")
        
        # Display suggestions
        print("\nSuggested Parameter Combinations:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(suggestions_df)
    else:
        print("Could not generate any new suggestions")


def run_cli():
    """
    Run the command-line interface for checkpoint management.
    """
    parser = argparse.ArgumentParser(description='Checkpoint Manager for Parameter Optimization')
    parser.add_argument('checkpoint_file', help='Path to the checkpoint file')
    parser.add_argument('--status', '-s', action='store_true', help='Show search status')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize parameter space coverage')
    parser.add_argument('--export', '-e', metavar='DIR', help='Export results to the specified directory')
    parser.add_argument('--suggest', '-g', metavar='N', type=int, default=0, 
                      help='Suggest N additional parameter combinations to explore')
    parser.add_argument('--output-dir', '-o', default=None, 
                      help='Output directory for visualizations and suggestions')
    parser.add_argument('--validate', '-t', action='store_true',
                      help='Run validation tests before execution')
    parser.add_argument('--force', '-f', action='store_true',
                      help='Force execution even if validation fails')
    
    args = parser.parse_args()
    
    # Default output directory
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"checkpoint_analysis_{timestamp}"
    
    # Run validation if requested
    if args.validate:
        validation_passed = validate_and_print(args.checkpoint_file, output_dir)
        if not validation_passed and not args.force:
            print("\nValidation failed. Use --force to run anyway.")
            return
        elif not validation_passed:
            print("\nProceeding despite validation failures...")
    
    # Create the checkpoint manager
    manager = CheckpointManager(args.checkpoint_file)
    
    # Process commands - order matters for a logical flow
    
    # First show status if requested
    if args.status:
        handle_status_command(manager)
    
    # Then handle visualization
    if args.visualize:
        handle_visualization_command(manager, output_dir)
    
    # Handle export
    if args.export:
        handle_export_command(manager, args.export)
    
    # Finally handle suggestions
    if args.suggest > 0:
        handle_suggest_command(manager, args.suggest, output_dir)
    
    # If no commands were given, show usage
    if not any([args.status, args.visualize, args.export, args.suggest > 0]):
        print("\nNo commands specified. Use one or more of these options:")
        print("  --status, -s: Show search status")
        print("  --visualize, -v: Visualize parameter space")
        print("  --export DIR, -e DIR: Export results to directory")
        print("  --suggest N, -g N: Suggest N new parameter combinations")
        print("\nFor more information, run with --help")