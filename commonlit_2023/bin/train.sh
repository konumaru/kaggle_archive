#!/bin/bash

python src/preprocessing.py && \
python src/feature.py && \
python src/train.py --multirun model=xgb,lgbm && \
python src/ensemble.py
