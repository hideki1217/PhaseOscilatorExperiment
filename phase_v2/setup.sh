#!/bin/bash

git clone https://github.com/pybind/pybind11.git
git clone https://github.com/nlohmann/json.git

wget https://boostorg.jfrog.io/artifactory/main/release/1.83.0/source/boost_1_83_0.tar.gz
tar -zxvf boost_1_83_0.tar.gz
rm boost_1_83_0.tar.gz

python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
