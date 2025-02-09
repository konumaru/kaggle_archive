#!/bin/bash

cp ./data/preprocessed/* ./data/upload/
cp -r data/model/xgb ./data/upload/
cp -r data/model/lgbm ./data/upload/
cp -r ./src/ ./data/upload/

kaggle datasets version -r "zip" -p ./data/upload/ -m "Update" 
