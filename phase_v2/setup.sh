#!/bin/bash

git clone https://github.com/pybind/pybind11.git
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
