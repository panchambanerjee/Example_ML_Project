#!/usr/bin/env bash

export TRAINING_DATA=input/train_folds.csv

export FOLD=0

python -m src.train
