#!/bin/bash
set -e

echo "Creating conda environment 'chemprop' with Python 3.11..."
conda create -n chemprop python=3.11 -y

echo "Installing Streamlit and pandas..."
# Using conda run so we don't have to source conda activate in this script
conda run -n chemprop pip install streamlit pandas

echo "Cloning Chemprop repository..."
if [ ! -d "chemprop" ]; then
    git clone https://github.com/chemprop/chemprop.git
else
    echo "chemprop directory already exists, skipping clone."
fi

echo "Installing Chemprop..."
cd chemprop
conda run -n chemprop pip install -e .
cd ..

echo "----------------------------------------------------"
echo "Setup complete! To run the web application, execute:"
echo "  conda activate chemprop"
echo "  streamlit run app.py"
echo "----------------------------------------------------"
