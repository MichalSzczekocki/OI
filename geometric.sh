#!/usr/bin/sh

TORCH='1.8.0+cu102'

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-geometric