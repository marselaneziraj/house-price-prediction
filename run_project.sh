#!/usr/bin/env bash
set -e
echo "Building processed dataset..."
python -m src.data.make_dataset
echo "Training models..."
python -m src.models.train
echo "Evaluating best model..."
python -m src.models.evaluate
