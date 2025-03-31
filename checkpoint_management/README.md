# Checkpoint Manager

A comprehensive tool for managing, analyzing, and visualizing parameter optimization checkpoints.

## Features

- Analyze the progress of parameter optimization runs
- Visualize parameter space exploration 
- Suggest promising new parameter combinations
- Validate checkpoints to ensure operations will succeed
- Modify pending parameter combinations

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/checkpoint-manager.git
   cd checkpoint-manager
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python checkpoint_manager.py CHECKPOINT_FILE [options]
```

### Options

- `--status, -s`: Show search status
- `--visualize, -v`: Visualize parameter space coverage
- `--export DIR, -e DIR`: Export results to directory
- `--suggest N, -g N`: Suggest N additional parameter combinations
- `--output-dir DIR, -o DIR`: Output directory (default: timestamped directory)
- `--validate, -t`: Run validation tests before execution
- `--force, -f`: Force execution even if validation fails

### Examples

Display the status of an optimization run:
```bash
python checkpoint_manager.py optimization_results/search_checkpoint.pkl --status
```

Visualize parameter space and suggest 5 new combinations:
```bash
python checkpoint_manager.py optimization_results/search_checkpoint.pkl --visualize --suggest 5
```

Run validation tests to check for potential errors:
```bash
python checkpoint_manager.py optimization_results/search_checkpoint.pkl --validate
```

Export results to a specific directory:
```bash
python checkpoint_manager.py optimization_results/search_checkpoint.pkl --export results_dir
```

### Test Script

A separate test script is available to validate a checkpoint file:

```bash
python test_checkpoint_manager.py optimization_results/search_checkpoint.pkl
```

## Module Structure

- `checkpoint_management/`: Main package
  - `manager.py`: Core checkpoint loading/saving
  - `visualization.py`: Data visualization functions
  - `suggestion_strategies.py`: Parameter suggestion algorithms
  - `param_utils.py`: Parameter type handling utilities
  - `validation.py`: Pre-execution validation functions
  - `cli.py`: Command-line interface

## Validation

The validation system tests:

1. Checkpoint structure and content
2. Parameter operations (type conversion, etc.)
3. Visualization functionality
4. Suggestion generation
5. Output directory accessibility

Running with `--validate` will perform these checks before execution to catch potential errors early.