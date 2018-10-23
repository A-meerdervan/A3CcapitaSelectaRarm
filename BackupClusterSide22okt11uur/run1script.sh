#!/bin/bash

# Name the job
#SBATCH -J pongS61run2hid64rollout40 
# This partition has the main partition
#SBATCH --partition=main
# A trick to set this partition exclusively to me
#SBATCH --cpus-per-task=17
# Say we will run on 1 machine, or one node
#SBATCH -N1
# Receive mail on start and finish
#SBATCH	--mail-type=ALL 

# run what you want to run
srun -N1 -n1 --exclusive pongS6run.sh &