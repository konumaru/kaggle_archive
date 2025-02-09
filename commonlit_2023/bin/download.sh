#!/bin/bash

set -ex pipefail

download_competition() {
    DIR_NAME=$1

    mkdir -p data/
    mkdir -p data/raw

    # Download competition data
    kaggle competitions download -c "$DIR_NAME"

    # Unzip
    unzip -o "$DIR_NAME.zip" -d "./data/raw"

    # Remove zip
    rm "$DIR_NAME.zip" 
}

download_dataset() {
    DATASET_NAME=$1
    IFS='/' read -ra ARR <<< "$DATASET_NAME"
    DIR_NAME="${ARR[1]}"

    mkdir -p data/
    mkdir -p data/external

    # Download dataset data
    kaggle datasets download -d "$DATASET_NAME"

    # Unzip
    unzip "$DIR_NAME.zip" -d "data/external/$DIR_NAME"

    # Remove zip
    rm "$DIR_NAME.zip"
}

download_kernels_dataset() {
    DATASET_NAME=$1
    IFS='/' read -ra ARR <<< "$DATASET_NAME"
    DIR_NAME="${ARR[1]}"

    mkdir -p data/
    mkdir -p data/external

    # Download dataset data
    kaggle kernels output $DATASET_NAME -p data/external/$DIR_NAME
}

# Download competitoin data.
download_competition commonlit-evaluate-student-summaries

# Download datasets.
# download_dataset konumaru/microsoft-deberta-v3-base

# Download kernels datasets.
# download_kernels_dataset konumaru/finetune-debertav3-training
download_kernels_dataset konumaru/finetune-debertav3-training-content
