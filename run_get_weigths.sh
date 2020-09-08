#!/bin/bash

# runs python script

## on argon
# module load tensorflow

mkdir weights

## using defaults parameters
# python3 get_weights.py 

## generating csv only for given file
python3 get_weights.py --model ../.yadlt/models/dae_model10.meta

# given parameters for output and directory of models (.meta files)
# parameters output and directory would need to be changed if names are different from the below line
# python3 get_weights.py --output ../weights/ --directory ../.yadlt/models/
