#!/bin/bash

# Name the job
#SBATCH -J pongS6runsNoSrun 
# This partition has the ctit80 node which is a nice new machine with 80 cpus 
#SBATCH --partition=main
# A trick to set this partition exclusively to me
#SBATCH --cpus-per-task=17
# Say we will run on 1 machine, or one node
#SBATCH -N1 # was N3 when it worked for 3 srun assingment
# Receive mail on start and finish
#SBATCH	--mail-type=ALL 

echo "in noSrun.sh"

cd A3CownAdaptToPong
#python3 mainA3C.py
python3 mainA3C.py