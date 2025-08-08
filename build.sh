#!/bin/bash

# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Run initial model training
# Note: We don't run populate_db.py here because we already did it manually.
python train_sklearn_model.py