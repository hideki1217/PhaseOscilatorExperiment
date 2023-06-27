#!/bin/bash

time ./phase.elf
time python3 ./phase.py
python3 ./notify.py
