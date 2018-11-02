# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 12:32:47 2018

@author: Alex
"""

import numpy as np
import csv
import gobalConst as cn
import matplotlib.pyplot as plt

# import from CSV
csvPath = 'CSVfromW031oktG98withEnv7.csv'
with open(csvPath, 'r') as f:
  reader = csv.reader(f)
  allLines = list(reader)
data = allLines[1:]
#1 getal = float((allLines[readI])[index])
totLines = len(allLines) - 1
dataToPlot = np.zeros([totLines,2])
for lineI in range(1,totLines + 1):
    dataToPlot[lineI - 1,0] = float((allLines[lineI])[1])
    dataToPlot[lineI - 1,1] = float((allLines[lineI])[2])
print(dataToPlot)
eps = dataToPlot[:,0]
lengths = dataToPlot[:,1]
# Plot the data
figName = "LengthsW0"
fig = plt.figure()
# Plot the score difference
plt.plot(eps, lengths, 'C7',label='worker0')
plt.ylabel("Length"); plt.xlabel("episode"); plt.legend()
plt.title("Length of worker 0") ; plt.show() 
# Save the fig to file
fig.savefig(figName + '.png')   

# Calculate the total amount of training states
#finalEp = (allLines[totLines])[0]
epStepSize = 5.
numOfWorkers = 12.
totalStatesWorker = epStepSize*np.sum(lengths)
totalStatesIn1e6 = (numOfWorkers * totalStatesWorker) / 1e6
print(totalStatesIn1e6)

