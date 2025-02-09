#!/bin/bash

MESSAGE=$(date '+%Y-%m-%d-%H-%M-%S')

rm  ./data/upload/*.txt
rm ./data/upload/*.pkl
rm -rf ./data/upload/src
rm -rf ./data/upload/xgb
rm -rf ./data/upload/cat
rm -rf ./data/upload/ridge

cp ./data/preprocessing/*.txt ./data/upload/
cp ./data/preprocessing/*.pkl ./data/upload/
cp -r ./data/train/* ./data/upload/
cp -r ./data/ensemble/* ./data/upload/
cp -r ./src/ ./data/upload/

rm  ./data/upload/*.parquet

kaggle datasets version -r "zip" -p ./data/upload/ -m $MESSAGE 
