#!/bin/bash

python src/feature.py && \
python src/train.py -m model=xgb,rf && \
python src/emsemble.py
