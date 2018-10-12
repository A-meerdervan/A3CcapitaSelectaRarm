# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:11:21 2018

@author: Alex
"""

import MyPong # My PyGame Pong Game
#import MyAgent # My DQN Based Agent
#import SelfplayAgent # previous version of DQN agent
#import numpy as np
#import random
#import matplotlib.pyplot as plt
from GlobalConstants import gc
#import SelfplayExperiment as spe
#import GamesManager as gm
#import CSVhandler as csvH
#import SimpleBot
import Simulator

def main():
    startIt = 0 # Normally 0
    nrOfCycles = 100 # Normally: = gc.NUMBER_CYCLES
    saveToCSV = True
    fromFile = gc.LOAD_MODELS_FROM_FILE
    sim = Simulator.Simulator(fromFile, gc.MODEL_FOLDER_PATH, gc.CSV_FILE_TRAIN)
    sim.runTrainingProcessBot(startIt, nrOfCycles, saveToCSV)
    #sim.runTrainingProcessSelfP(startIt, nrOfCycles, saveToCSV)
    sim.saveAgentsToFile() 

if __name__ == "__main__":
    try:
        main()
        if gc.RENDER_SCREEN: MyPong.quit_()
    except:
        print("Unexpected error")
        #Quit the pygame window
        if gc.RENDER_SCREEN: MyPong.quit_()
        raise


