#!/bin/bash

# Name the job
#SBATCH -J pongS6runs 
# This partition has the ctit80 node which is a nice new machine with 80 cpus 
#SBATCH --partition=r730 
# A trick to set this partition exclusively to me
#SBATCH --cpus-per-task=80 
# Say we will run on 1 machine, or one node
#SBATCH -N1 
# Receive mail on start and finish
#SBATCH	--mail-type=ALL 

# run what you want to run
srun -N1 pongS6run.sh
#srun -N1 pongS6run2.sh
