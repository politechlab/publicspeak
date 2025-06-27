# PublicSpeak PSL Model Package

This package contains the PSL (Probabilistic Soft Logic) models for the PublicSpeak project, used for public comment classification.

## Package Structure

```
publicspeak/model/
├── __init__.py          # Package initialization file
├── main.py              # Unified entry point
├── training/            # Training module
│   ├── __init__.py
│   └── train.py
├── inference/           # Inference module
│   ├── __init__.py
│   └── infer.py
└── paper_reproduce/     # Paper reproduction module
    ├── __init__.py
    ├── infer.py
    └── weight_file.json
```

## Usage

### 1. Unified Entry Point (Recommended)

Run from the `publicspeak` directory:

```bash
# Train model (uses fixed data paths)
python -m model.main --mode train --output output

# Run inference (uses fixed data paths)
python -m model.main --mode infer --output output

# Run paper reproduction (uses fixed data paths)
python -m model.main --mode paper_reproduce --output output

# Use custom parameters
python -m model.main --mode train --seed 123 --output output
```

### 2. Direct Module Execution

```bash
# Training (no city parameter needed)
python -m model.training.train

# Inference (no city parameter needed)
python -m model.inference.infer

# Paper reproduction (no city parameter needed)
python -m model.paper_reproduce.infer
```

### 3. Import in Code

```python
from publicspeak.model import train_main, infer_main, paper_infer_main
import argparse

# Create args object
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='output')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args([])

# Training (pass args object)
train_main(args)

# Inference (pass args object)
infer_main(args)

# Paper reproduction (pass args object)
paper_infer_main(args)
```

## Parameters

- `--mode`: Operation mode, `train`, `infer`, or `paper_reproduce`
- `--output`: Output directory, defaults to `output`
- `--seed`: Random seed, defaults to `42`

## Configuration

Model configuration is defined in `config/settings.py` and `config/paths.py`:

- `Settings.PSL_TRAIN_MODEL_NAME`: Training model name
- `Settings.PSL_TEST_MODEL_NAME`: Test model name
- `Paths.PSL_GENERATED_TRAIN_DATA`: Fixed training data directory
- `Paths.PSL_PROCESSED_TEST_DATA`: Fixed test data directory

## Dependencies

- PSL (Probabilistic Soft Logic) Python package
- NumPy
- scikit-learn
- Other dependencies as specified in the project requirements

## Notes

1. Make sure to run commands from the `publicspeak` directory
2. Ensure all required data files exist in the configured directories
3. Make sure `init_weight_file.json` exists before training
4. Make sure `weight_file.json` exists before inference
5. Both training and inference use fixed data paths and don't require city specification
6. Paper reproduction module uses its own weight file in the `paper_reproduce` directory

## Examples

### Training Example
```bash
# Train model with default settings (no city needed)
python -m model.main --mode train

# Train model with custom seed
python -m model.main --mode train --seed 123 --output results
```

### Inference Example
```bash
# Run inference with default settings (no city needed)
python -m model.main --mode infer

# Run inference with custom output directory
python -m model.main --mode infer --output results
```

### Paper Reproduction Example
```bash
# Run paper reproduction with default settings
python -m model.main --mode paper_reproduce

# Run paper reproduction with custom output directory
python -m model.main --mode paper_reproduce --output results
```

Dependencies:

- LINK TO PSL 
  
