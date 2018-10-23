#!/bin/bash

# Name the job
#SBATCH -J pongS6runs 
# This partition has the ctit80 node which is a nice new machine with 80 cpus 
#SBATCH --partition=main
# A trick to set this partition exclusively to me
#SBATCH --cpus-per-task=17
# Say we will run on 1 machine, or one node
#SBATCH -N3 # was N3 when it worked for 3 srun assingment
# Receive mail on start and finish
#SBATCH	--mail-type=ALL 

# run what you want to run
srun -N1 -n1 --exclusive pongS6run1.sh &
srun -N1 -n1 --exclusive pongS6run2.sh &
srun -N1 -n1 --exclusive pongS6run3.sh 