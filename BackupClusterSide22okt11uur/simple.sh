#!/bin/bash

#SBATCH -N1
#SBATCH -J simple --mail-type=ALL

echo "Hello World!"

cd A3CownAdaptToPong
python3 evalModel.py
