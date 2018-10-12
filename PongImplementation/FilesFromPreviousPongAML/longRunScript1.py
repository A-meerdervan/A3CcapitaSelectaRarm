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
#import GlobalConstants as gc
from GlobalConstants import gc
#import SelfplayExperiment as spe
#import GamesManager as gm
#import CSVhandler as csvH
#import SimpleBot
import Simulator
import pathlib # Used to create folders

def runTrainingSession(modelType, playBot, nrOfCycles, trainTime, saveToCSV, modelsFolder, trainBot, numTestGames, numTrainOpp, trainInterval, Qinterval):
    gc.resetModels()
    startIt = 0 # Normally 0
    fromFile = gc.LOAD_MODELS_FROM_FILE
    gc.setModelType(modelType)
    gc.setNrOfCycles(nrOfCycles)
    gc.setTrainTime(trainTime)
    gc.setModelsFolder(modelsFolder)
    gc.setTrainBotType(trainBot)
    gc.setNofTestGames(numTestGames)
    gc.setNofTrainOpp(numTrainOpp)
    gc.setTrainInterval(trainInterval)
    gc.setQ_NN_updateInterval(Qinterval)
    sim = Simulator.Simulator(fromFile, modelsFolder, gc.CSV_FILE_TRAIN)
    if playBot:
        sim.runTrainingProcessBot(startIt, gc.NUMBER_CYCLES, saveToCSV)
    else:
        sim.runTrainingProcessSelfP(startIt, gc.NUMBER_CYCLES, saveToCSV)
    # Save the brains array if not working from files
    # if not fromFile: sim.saveAgentsToFile() 

def main():
    # Troughout
    topFolder = "1febEve_99gamma_SelfPlay_AndBot_TT_500000"
    saveToCSV = True
    # CRASHES AT 1!
    cyclesList =    [20,20]#,100,100,100,100]#[100,100,100,100] + [100,100,100,100] + [100,100,100,100] # Normally gc.NUMBER_CYCLES
    numOfRuns = len(cyclesList)
    trainTimes =    [500000,500000,100000,100000,100000,100000] #[20000,20000,20000,20000] + [20000,20000,20000,20000] + [20000,20000,20000,20000]
    playBotList =   [False,True,True,True,True,True] #[False,False,False,False] + [False,False,False,False] + [True,True,True,True]
    trainBots =     [10,10,10,10,10]#[-1,-1,-1,-1] + [-1,-1,-1,-1]  + [9,9,9,9] # -1 per SELFPLAY run (playBot = False)
    numTrainGames = [3,-1,-1,-1,-1,-1] #[3,3,3,3] + [3,3,3,3] + [-1,-1,-1,-1] # -1 per BOT run (playBot = true)
    numTestGames =  [10,10,10,10,10,10]   #[5,5,5,5] + [5,5,5,5] + [5,5,5,5]
    modelTypes = [5,5,5,5,5,5] #[1,2,3,4] + [1,2,3,4] + [1,2,3,4] #,3,4,1,2,3,4]
    trainIntervals = [10,10,10,10,10,10] #[10,10,10,10] + [3,3,3,3] + [10,10,10,10] #,10]
    Qintervals = [200,200] #,200,300,400,1000]

    for i in range(numOfRuns):        
        modelsFolder = topFolder + "/models" + str(i)
        pathlib.Path(modelsFolder).mkdir(parents=True, exist_ok=True) 
        runTrainingSession(modelTypes[i], playBotList[i], cyclesList[i], trainTimes[i], saveToCSV, modelsFolder, trainBots[i], numTestGames[i],numTrainGames[i], trainIntervals[i], Qintervals[i])

if __name__ == "__main__":
    try:
        main()
        
        if gc.RENDER_SCREEN: MyPong.quit_()
    except:
        print("Unexpected error")
        #Quit the pygame window
        if gc.RENDER_SCREEN: MyPong.quit_()
        raise


