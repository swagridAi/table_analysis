#!/usr/bin/env python3
"""
Main entry point for the checkpoint management system.

This script provides a command-line interface for the checkpoint management system,
allowing users to analyze parameter optimization checkpoints.
"""

from checkpoint_management.cli import run_cli


if __name__ == "__main__":
    run_cli()