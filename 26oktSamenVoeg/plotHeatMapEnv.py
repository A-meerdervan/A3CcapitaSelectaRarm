import threading
#import multiprocessingq
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import gobalConst as cn
#from skimage.color import rgb2gray # gave an error, had to pip it, then another error then had to pip scikit-image
#matplotlib inline
#from helper import * # had to pip it
#from vizdoom import *
#import gym # had to pip it and then had to pip gym[atari], then had to pip make, but better from right to left :P!
import SimulationEnvironment as sim
import Env.A3CenvPong # import custom Pong env (no images but 6 numbers as state)
import pygame
#from random import choice
#from time import sleep
from time import time
import os
#import logging # logs messages in multithreading applications

import numpy as np
import csv
import gobalConst as cn
import matplotlib.pyplot as plt


def pltLine(p1,p2):
    plt.plot([p1[0],p2[0]],[p1[1],p2[1]],linestyle= (0, ()), linewidth=4, color='red')
def plotWalls(envWalls):
    for wall in envWalls:
        p1 = wall[0]
        p2 = wall[1]
        pltLine(p1,p2)
# Take a set of connected points that go clockwise from left bottom to right bottom
# and convert it to a valid wall construct. points = np.array([(x1,y1),(x2,y2),...])
def getWalls(points):
    walls = []
    startPoint = points[0]; curEndPoint = startPoint
    for point in points[1:]:
        walls.append([curEndPoint,point])
        curEndPoint = point
    walls.append([points[-1],startPoint])
    return np.array(walls)


env = sim.SimulationEnvironment()
#env.runTestMode(cn.TEST_fromTestConsts)
envNr = 1
totalDots = 20000
Xpts,Ypts = env.createHeatMap(envNr,totalDots)
envWalls = env.envWalls
# plot heatmap:
figName = "HeatmapEndEffector_Report"
fig = plt.figure()
plt.scatter(Xpts, Ypts,s=1)
plotWalls(envWalls)
# Plot a line which shows the lower Y goal threshold
#plt.plot([0,400],[280,280],linestyle= (0, ()), linewidth=4, color='black')
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
#plt.set_xlim(0, 400)
plt.xlim([0,400])
plt.ylim([400, 0])
plt.axis('equal')
plt.grid(True)
plt.ylabel("y in pixels"); plt.xlabel("x in pixels"); #plt.legend()
plt.title("end effector heatmap for envNr = " + str(envNr)) ; plt.show()
# Save the fig to file
fig.savefig(figName + '.png')
#
##plt.scatter(Xpts, Ypts, 'C7',label='endEffector')
