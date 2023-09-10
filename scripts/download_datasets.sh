#!/bin/bash
# Downlad the datasets from google drive if not already present
if [ ! -d ./dataset ]; then
    pip freeze | grep "^gdown" || pip install gdown
    gdown "1alE33S1GmP5wACMXaLu50rDIoVzBM4ik"
    unzip -n -j all_six_datasets.zip -d dataset
    rm dataset/.*
    rm all_six_datasets.zip
fi
