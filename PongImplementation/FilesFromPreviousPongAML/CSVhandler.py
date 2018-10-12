# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:54:27 2018

@author: Alex
"""

#import MyPong # My PyGame Pong Game
#import MyAgent # My DQN Based Agent
#import SimpleBot
#import SelfplayAgent # previous version of DQN agent
#import random
#import matplotlib.pyplot as plt
#import GlobalConstants as gc

import numpy as np

def saveListToCSV(filepath, _list):
    with open(filepath,'ab') as f:
        np.savetxt(f, [_list], delimiter=',', fmt='%f')