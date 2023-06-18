#!/bin/bash

./phase > phase_swap.csv
python3 ./phase.py
python3 ./phase_K.py 13
