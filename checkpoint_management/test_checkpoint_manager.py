#!/usr/bin/env python3
"""
Test script for the checkpoint management system.

This script runs validation tests on a checkpoint file to ensure
all operations will work correctly before executing them.
"""

import sys
import argparse
from checkpoint_management.validation import validate_and_print


def main():
    """Main function to run validation tests."""
    parser = argparse.ArgumentParser(description='Test Checkpoint Manager Functionality')
    parser.add_argument('checkpoint_file', help='Path to the checkpoint file')
    parser.add_argument('--output-dir', '-o', default='test_output', 
                      help='Output directory for test files')
    
    args = parser.parse_args()
    
    # Run validation tests and exit with appropriate status code
    validation_passed = validate_and_print(args.checkpoint_file, args.output_dir)
    
    # Return exit code based on validation result
    sys.exit(0 if validation_passed else 1)


if __name__ == "__main__":
    main()