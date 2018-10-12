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

def runTestGamesOverFolders():
    # General
    fromFile = True # Not gc.LOAD_MODELS_FROM_FILE because when testing you
    # always read models from file
    saveToCSV = True
    # Number of test games between two opponents
    numGames = 10
    folders = []
    numRuns = []
    paramsPerFolder = []
    # Per folder
#    folders.append("../PongCode30janOchtendPCQupdate/29janNachtRunQintervalTest/")
#    # The parameters from the file that run it
#    modelTypes = [2,2,2,2,2] + [1,2,3,4,5] #+ [1,2,3,4] #,3,4,1,2,3,4]
#    playBotList =   [False,False,False,False,False] + [True,True,True,True,True] #+ [True,True,True,True]
#    cyclesList =    [30,30,30,30,30] + [50,50,50,50,50] #+ [50,50,50,50,50] # Normally gc.NUMBER_CYCLES
#    numRuns.append(len(modelTypes))
#    trainTimes =    [20000,20000,20000,20000,20000] + [20000,20000,20000,20000,20000] #+ [20000,20000,20000,20000]
#    trainBots =     [-1,-1,-1,-1,-1] + [10,10,10,10,10] # + [9,9,9,9] # -1 per SELFPLAY run (playBot = False)
#    numTestGames =  [10,10,10,10,10] + [10,10,10,10,10] #+ [10,10,10,10]
#    numTrainGames = [3,3,3,3,3] + [-1,-1,-1,-1,-1] # -1 per BOT run (playBot = true)
#    trainIntervals = [10,10,10,10,10] + [10,10,10,10,10] #+ [10,10,10,10] #,10]
#    Qintervals = [100,150,200,400,1000] + [200,200,200,200,200]
#    # Add to the all of it to the list
#    paramsPerFolder.append([modelTypes, playBotList, cyclesList, trainTimes, trainBots, numTestGames, numTrainGames, trainIntervals, Qintervals])
    
#    folders.append("../pongCode29janAvondPC/largeRuns/")
#    # The parameters from the file that run it
#    modelTypes = [1,2,3,4] + [1,2,3]
#    playBotList =  [False,False,False,False] + [False,False,False]
#    cyclesList =   [100,100,100,100] + [100,100,100]
#    numRuns.append(len(modelTypes))
#    trainTimes =   [20000,20000,20000,20000] + [20000,20000,20000]
#    trainBots =  [-1,-1,-1,-1] +             [-1,-1,-1,-1] 
#    numTestGames =   [10,10,10,10] + [10,10,10,10]
#    numTrainGames = [3,3,3,3] + [3,3,3,3]
#    trainIntervals = [10,10,10,10] + [3,3,3,3]
#    Qintervals = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#    # Add to the all of it to the list
#    paramsPerFolder.append([modelTypes, playBotList, cyclesList, trainTimes, trainBots, numTestGames, numTrainGames, trainIntervals, Qintervals])
    
    
#    folders.append("1febLongTrainRun_100k_BOT_andSP/")
#    # The parameters from the file that run it
#    modelTypes = [5,5]
#    playBotList =  [False,True]
#    cyclesList =   [30,30]
#    numRuns.append(len(modelTypes))
#    trainTimes =   [100000,100000]
#    trainBots =  [-1,-1,-1,-1] +             [-1,-1,-1,-1] 
#    numTestGames =   [10,10,10,10] + [10,10,10,10]
#    numTrainGames = [3,3,3,3] + [3,3,3,3]
#    trainIntervals = [10,10,10,10] + [3,3,3,3]
#    Qintervals = [200,200,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#    # Add to the all of it to the list
#    paramsPerFolder.append([modelTypes, playBotList, cyclesList, trainTimes, trainBots, numTestGames, numTrainGames, trainIntervals, Qintervals])
    
#    folders.append("1febEve_99gamma_100k_QintervalsBOT/")
#    # The parameters from the file that run it
#    modelTypes = [5,5,5,5,5]
#    playBotList =   [True,True,True,True,True,True]
#    cyclesList =    [30,30,30,30,30]
#    numRuns.append(len(modelTypes))
#    trainTimes =    [100000,100000,100000,100000,100000,100000]
#    trainBots =     [10,10,10,10,10]
#    numTestGames =   [10,10,10,10] + [10,10,10,10]
#    numTrainGames = [-1,-1,-1,-1,-1,-1]
#    trainIntervals = [10,10,10,10,10,10]
#    Qintervals = [0,150,200,300,400]
#    # gamma = .99
#    # Add to the all of it to the list
#    paramsPerFolder.append([modelTypes, playBotList, cyclesList, trainTimes, trainBots, numTestGames, numTrainGames, trainIntervals, Qintervals])  
    
    folders.append("1febEve_99gamma_SelfPlay_AndBot_TT_500000/")
    # The parameters from the file that run it
    modelTypes = [5] #5,5,5,5]
    playBotList =   [False]#0True,True,True,True,True]
    cyclesList =    [18]#,30,30,30,30]
    numRuns.append(len(modelTypes))
    trainTimes =    [500000]#,100000,100000,100000,100000,100000]
    trainBots =     [10]#10,10,10,10]
    numTestGames =   [10]#10,10,10] + [10,10,10,10]
    numTrainGames = [-1]#,-1,-1,-1,-1,-1]
    trainIntervals = [10]#,10,10,10,10,10]
    Qintervals = [200,200]#,200,300,400]
    # gamma = .99
    # Add to the all of it to the list
    paramsPerFolder.append([modelTypes, playBotList, cyclesList, trainTimes, trainBots, numTestGames, numTrainGames, trainIntervals, Qintervals])  
    
    
    # Loop all folders
    for f in range( len(folders) ):
        folderPath = folders[f]
        params = paramsPerFolder[f]
        for run in range(numRuns[f]):
            modelsPath = folderPath + "models" + str(run)
            setRelevantGlobals(params,run)
            # The last version (best version hopefully)
            iterOpponent = (params[2])[run]
            playBot = False
            # Loop for best against it
            for it in range(1,iterOpponent + 1, 2):
                runTestGames(numGames, fromFile, it, playBot, iterOpponent, -1, modelsPath, saveToCSV)
            # All loops hereafter go against a bot
            playBot = True
            # Loop for bot 13 against it
            for it in range(1,iterOpponent + 1, 2):
                runTestGames(numGames, fromFile, it, playBot, -1, 13, modelsPath, saveToCSV)
            # Loop for bot 10 against it
            for it in range(1,iterOpponent + 1, 2):
                runTestGames(numGames, fromFile, it, playBot, -1, 10, modelsPath, saveToCSV)
#            # Loop for bot 8 against it
#            for it in range(1,iterOpponent + 1, 2):
#                runTestGames(numGames, fromFile, it, playBot, -1, 8, modelsPath, saveToCSV)
            # Loop for bot 6 against it
            for it in range(1,iterOpponent + 1, 2):
                runTestGames(numGames, fromFile, it, playBot, -1, 6, modelsPath, saveToCSV)

def setRelevantGlobals(params, run):
    print(run)
    gc.resetModels()
    gc.setModelType((params[0])[run])
    gc.setNrOfCycles((params[1])[run])
    #gc.setTrainTime((params[2])[run])
    #gc.setModelsFolder((params[0])[run])
    #gc.setTrainBotType((params[0])[run])
    #gc.setNofTestGames((params[0])[run])
    #gc.setNofTrainOpp((params[0])[run])
    #gc.setTrainInterval((params[0])[run])
    #gc.setQ_NN_updateInterval((params[0])[run])
    
        
# BrainToTest is de iteration die je test
# iterOpponent is de beste of -1 wanneer het een bot is
def runTestGames(nrOfGames, fromFile, brainToTest, playBot, iterOpponent, botType, modelsPath, saveToCSV):
    sim = Simulator.Simulator(fromFile, modelsPath, gc.CSV_FILE_TEST) #gc.TEST_MODEL_FOLDER_PATH, gc.CSV_FILE_TEST)
    iteration = -10
    if playBot:
        sim.runBotTestGames(botType, brainToTest, nrOfGames, saveToCSV, iteration)
        #sim.runBotTestGames(botType, brainToTest, gc.N_GAMES_NO_TRAIN)    
    else:
        sim.runSelfPlayTestGames(iterOpponent,brainToTest, nrOfGames,saveToCSV, iteration) 
    
# This function temporaryly holds all to run test games
def runTestGamesSolo():
    # General
    #modelsPath = "../PongCode30janOchtendPCQupdate/29janNachtRunQintervalTest/models8"
    modelsPath = "1febEve_99gamma_100k_MT5_/models0" # number 5 is epic!
    #modelsPath = "1febEve_99gamma_100k_QintervalsBOT/models4"
    #modelsPath = "1febLongTrainRun_100k_BOT_andSP/models0"
    #modelsPath = "../pongCode29janAvondPC/largeRuns/models1"    
    #gc.setModelType(2)
    nrOfGames = 10 #gc.N_GAMES_NO_TRAIN
    playBot = True
    brainToTest = 5
    saveToCSV = False
    # If against bot
    botType = 10 # 4 is the best bot, 0 the worst
    # If Self play
    iterOpponent = 1
    fromFile = True # Not gc.LOAD_MODELS_FROM_FILE because when testing you
    # always read models from file
    runTestGames(nrOfGames, fromFile, brainToTest, playBot, iterOpponent, botType, modelsPath, saveToCSV)
    
#def main():
#    # Troughout
#    topFolder = "29janNachtRunQintervalTest"
#    saveToCSV = True
#    # CRASHES AT 1!
#    cyclesList =    [100,100,100,100,100]#[100,100,100,100] + [100,100,100,100] + [100,100,100,100] # Normally gc.NUMBER_CYCLES
#    numOfRuns = len(cyclesList)
#    trainTimes =    [20000,20000,20000,20000,20000] #[20000,20000,20000,20000] + [20000,20000,20000,20000] + [20000,20000,20000,20000]
#    playBotList =   [False,False,False,False,False] #[False,False,False,False] + [False,False,False,False] + [True,True,True,True]
#    trainBots =     [-1,-1,-1,-1,-1]#[-1,-1,-1,-1] + [-1,-1,-1,-1]  + [9,9,9,9] # -1 per SELFPLAY run (playBot = False)
#    numTrainGames = [3,3,3,3,3] #[3,3,3,3] + [3,3,3,3] + [-1,-1,-1,-1] # -1 per BOT run (playBot = true)
#    numTestGames =  [10,10,10,10,10]   #[5,5,5,5] + [5,5,5,5] + [5,5,5,5]
#    modelTypes = [1,1,1,1,1] #[1,2,3,4] + [1,2,3,4] + [1,2,3,4] #,3,4,1,2,3,4]
#    trainIntervals = [10,10,10,10,10] #[10,10,10,10] + [3,3,3,3] + [10,10,10,10] #,10]
#    Qintervals = [100,150,200,300,400]
#
#    for i in range(numOfRuns):        
#        modelsFolder = topFolder + "/models" + str(i)
#        pathlib.Path(modelsFolder).mkdir(parents=True, exist_ok=True) 
#        runTrainingSession(modelTypes[i], playBotList[i], cyclesList[i], trainTimes[i], saveToCSV, modelsFolder, trainBots[i], numTestGames[i],numTrainGames[i], trainIntervals[i], Qintervals[i])
#    
  
def main():
    #runTestGamesOverFolders()
    runTestGamesSolo()

if __name__ == "__main__":
    try:
        main()
        if gc.RENDER_SCREEN: MyPong.quit_()
    except:
        print("Unexpected error")
        #Quit the pygame window
        if gc.RENDER_SCREEN: MyPong.quit_()
        raise


