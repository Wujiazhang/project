#!/bin/bash
#SBATCH -N 1
#SBATCH -p c01
#SBATCH -n 6
#SBATCH -o test.log

python test.py

