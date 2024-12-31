#!/bin/bash

export TRANSFORMERS_CACHE=../cache

cd ../examples/image_classification
CUDA_VISIBLE_DEVICES=0 python main.py

