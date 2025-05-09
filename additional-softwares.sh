#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e


# Install PyTorch (CPU-only version, change below if you want CUDA support)
echo "Installing PyTorch..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
echo "Installing torch-spline-conv..."
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
echo "Installing torch-sparse..."
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
echo "Installing torch-cluster..."
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
echo "Installing torch-geometric..."
pip install torch-geometric==2.0.1
echo "Installing torch-scatter..."
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html


echo "PyTorch installation complete!"


pip install flask

pip install gunicorn