# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 13:21:50 2018

@author: Alex
"""

import MyPong # My PyGame Pong Game
import MyAgent # My DQN Based Agent
import SelfplayAgent # previous version of DQN agent
import numpy as np
import random
#import matplotlib.pyplot as plt
from GlobalConstants import gc
#import SelfplayExperiment as spe
import GamesManager as gm
import CSVhandler as csvH

import SimpleBot

    # =======================================================================
    
class Simulator:
    
    def __init__(self, fromFile, modelsFolderPath, csvFileName):
        self.theAgent = MyAgent.Agent(gc.STATECOUNT,gc.ACTIONS)
        self.renderScreen = gc.RENDER_SCREEN
        self.modelsPath = modelsFolderPath
        self.csvPath = modelsFolderPath + "/" + csvFileName
        self.fromFile = fromFile

    def runBotTestGames(self, botType, brainToTest, nrOfGames, saveToCSV, iteration):
        self.theAgent.LoadBrain(self.fromFile, self.modelsPath, brainToTest)
        oppAgent = SimpleBot.SimpleBot(botType)
        self.runTestGames(oppAgent, nrOfGames, saveToCSV, iteration)
        
    def runSelfPlayTestGames(self, iterOpponent, brainToTest, nrOfGames, saveToCSV, iteration):
        self.theAgent.LoadBrain(self.fromFile, self.modelsPath, brainToTest)
        oppAgent = SelfplayAgent.SelfplayAgent(gc.STATECOUNT, gc.ACTIONS, self.fromFile, self.modelsPath, iterOpponent) 
        self.runTestGames(oppAgent, nrOfGames, saveToCSV, iteration)
        
    def runRandomOppTestGames(self, brainToTest, startBrain, nrOfGames, saveToCSV, iteration):
        self.theAgent.LoadBrain(self.fromFile, self.modelsPath, brainToTest)
        agentTotalScore = 0
        oppTotalScore = 0
        wins = 0
        # This line will be put in the CSV file
        toCSV = np.append([0, 0, iteration, startBrain], np.zeros(nrOfGames + 3))
        for tg in range(nrOfGames):
            # Initiate the opponent random from the history
            # TODO Only sample opponents from the ones that were succesful
            # Using startBrain makes sure that only previous opponents are used.
            iterOpponent = random.randint(0, startBrain)
            toCSV[4 + tg] = iterOpponent
            OpponentAgent = SelfplayAgent.SelfplayAgent(gc.STATECOUNT, gc.ACTIONS, self.fromFile, self.modelsPath, iterOpponent) 
            [scoreAI, scoreOld, hasWon] = gm.PlayGame(self.theAgent, OpponentAgent, self.renderScreen, saveToCSV, self.csvPath)            
            agentTotalScore += scoreAI
            oppTotalScore += scoreOld
            wins += hasWon
        # Finish the toCSV line with the number of wins
        toCSV[4 + gc.N_of_Test_Games] = wins
        toCSV[4 + gc.N_of_Test_Games + 1] = agentTotalScore
        toCSV[4 + gc.N_of_Test_Games + 2] = oppTotalScore
        print(toCSV)
        if saveToCSV: csvH.saveListToCSV(self.csvPath, toCSV)
        winLose = "WON" if (agentTotalScore > oppTotalScore) else "LOST"
        print("The AI ", winLose, ", stats: AgentTscore, ", agentTotalScore, " OppTscore ", oppTotalScore, " wins ", wins, " out of ", nrOfGames)
        return [agentTotalScore, oppTotalScore]
    
    def runTestGames(self, oppAgent, nrOfGames, saveToCSV, iteration):
            print("=========================================")
            print("\n start game of agent with brain ", self.theAgent.brainNr," \n")
            print("Against opponent with brain       ", oppAgent.brainNr, " of type ", oppAgent.type)
            agentTotalScore = 0
            oppTotalScore = 0
            wins = 0
            # This line will be put in the CSV file
            toCSV = np.append([0, 0, iteration, self.theAgent.brainNr, oppAgent.brainNr, oppAgent.type], np.zeros(3))
            for tg in range(nrOfGames):
                [scoreAI, scoreOld, hasWon] = gm.PlayGame(self.theAgent, oppAgent, self.renderScreen, saveToCSV, self.csvPath)
                print("subResults",scoreAI, scoreOld, hasWon)
                agentTotalScore += scoreAI
                oppTotalScore += scoreOld
                wins += hasWon
            # Finish the toCSV line with the number of wins and such
            toCSV[6] = wins
            toCSV[7] = agentTotalScore
            toCSV[8] = oppTotalScore
            print(toCSV)
            if saveToCSV: csvH.saveListToCSV(self.csvPath, toCSV)
            # If the new AI player is better then the old players
            winLose = "WON" if (agentTotalScore > oppTotalScore) else "LOST"
            print("The AI ", winLose, ", stats: AgentTscore, ", agentTotalScore, " OppTscore ", oppTotalScore, " wins ", wins, " out of ", nrOfGames)
    
    def trainAgainstBotOnce(self, agentBrainNr, botType, iteration):
        #  Load our Agent (including DQN based Brain)
        self.theAgent.LoadBrain(self.fromFile, self.modelsPath, agentBrainNr)
        oppAgent = SimpleBot.SimpleBot(botType)
        gm.TrainAgent(self.theAgent, oppAgent, self.renderScreen, iteration)
        # After training, save the trained model
        #self.theAgent.SaveBrainWeights(self.modelsPath, iteration)   
        self.theAgent.SaveBrain()
            
    def trainAgainstPrevSelfs(self, agentBrainNr, iteration):
        #  Load our Agent (including DQN based Brain)
        self.theAgent.LoadBrain(self.fromFile, self.modelsPath, agentBrainNr)
        #TODO maybe only do this from iteration 2 or something?
        #Or when number of iterations > number of oppents
        for opp in range(gc.N_Of_Train_Oppnts):
            # Initiate the opponent random from the history
            # TODO Only sample opponents from the ones that were succesful
            iterOpponent = random.randint(0, agentBrainNr)
            oppAgent = SelfplayAgent.SelfplayAgent(gc.STATECOUNT, gc.ACTIONS, self.fromFile, self.modelsPath, iterOpponent) 
            print("=========================================")
            print("\n start training the agent against opponent, ", opp, "with brain: ", iterOpponent, "\n")
            gm.TrainAgent(self.theAgent, oppAgent, self.renderScreen, iteration)
        # After training multiple opponents, save the trained model
        #self.theAgent.SaveBrainWeights(self.modelsPath, iteration)   
        self.theAgent.SaveBrain()
    
    def runTrainingProcessSelfP(self, startIt, nrOfCycles, saveToCSV):
        startBrain = startIt
        # Normally run from 1, else from startBrain + 1
        for iteration in range(startBrain + 1, nrOfCycles):
            print("\n start training agent with brain: ", startBrain)
            self.trainAgainstPrevSelfs(startBrain, iteration)     
            print("\n starting competitive games\n")
            [agentTotalScore, oppTotalScore] = self.runRandomOppTestGames(iteration, startBrain, gc.N_of_Test_Games, saveToCSV, iteration)
            # if the new AI player is better then the old players, then use its brain next time
            if (agentTotalScore > oppTotalScore): startBrain = iteration            
    
    def runTrainingProcessBot(self, startIt, nrOfCycles, saveToCSV):
        startBrain = startIt
        for iteration in range(startBrain + 1, nrOfCycles):
            print("\n start training agent with brain: ", startBrain)
            self.trainAgainstBotOnce(startBrain, gc.TRAIN_BOT_TYPE, iteration)
            print("\n starting competitive games\n")
            self.runBotTestGames(gc.TRAIN_BOT_TYPE, iteration, gc.N_of_Test_Games, saveToCSV, iteration)
            # When training against the bot, always use the last brain
            startBrain = iteration  
    # Save a brain to a file in one go
    def saveAgentsToFile(self):
        self.theAgent.SaveBrainWeights(self.modelsPath)

# This function temporaryly holds all to run test games
def runTestGames():
    # General
    nrOfGames = 1#gc.N_GAMES_NO_TRAIN
    playBot = False
    brainToTest = 100
    saveToCSV = False
    # If against bot
    botType = 7 # 4 is the best bot, 0 the worst
    # If Self play
    iterOpponent = 100
    fromFile = True #gc.LOAD_MODELS_FROM_FILE
    
    sim = Simulator(fromFile, gc.TEST_MODEL_FOLDER_PATH, gc.CSV_FILE_TEST)
    print("From folder ", sim.modelsPath)
    iteration = -10
    if playBot:
        sim.runBotTestGames(botType, brainToTest, nrOfGames, saveToCSV, iteration)
        #sim.runBotTestGames(botType, brainToTest, gc.N_GAMES_NO_TRAIN)    
    else:
        sim.runSelfPlayTestGames(iterOpponent,brainToTest, nrOfGames,saveToCSV, iteration)
  
#def main():
#    runTestGames()

def main():
    startIt = 0 # Normally 0
    nrOfCycles = 100 # Normally: = gc.NUMBER_CYCLES
    saveToCSV = True
    fromFile = gc.LOAD_MODELS_FROM_FILE
    sim = Simulator(fromFile, gc.MODEL_FOLDER_PATH, gc.CSV_FILE_TRAIN)
    sim.runTrainingProcessBot(startIt, nrOfCycles, saveToCSV)
    #sim.runTrainingProcessSelfP(startIt, nrOfCycles, saveToCSV)
    sim.saveAgentsToFile()
 
    
    
### TODO REMOVE THIS:
#def main():
#    # TODO make this 0 again, now 49 to build on previous run
#    iterationLastSucces = 0 # = 0
#    renderScreen = True
#    # TODO have a nice comment here
#    for iteration in range(gc.NUMBER_CYCLES):
#    #for iteration in range(iterationLastSucces,gc.NUMBER_CYCLES*3):
#        #TODO remove (was om tegen bot te runnen)
#        iterationLastSucces = iteration
#        #  Create our Agent (including DQN based Brain)
#        TheAgent = MyAgent.Agent(gc.STATECOUNT,gc.ACTIONS)
#        TheAgent.LoadBrain(gc.MODEL_FOLDER_PATH, iterationLastSucces)
#        print("=========================================")
#        print("\n start training the agent with brain: ",iterationLastSucces," \n")
#        #TODO maybe only do this from iteration 2 or something?
#        # Or when number of iterations > number of oppents
##        for opp in range(gc.N_Of_Train_Oppnts):
##            # Initiate the opponent random from the history
##            # TODO Only sample opponents from the ones that were succesful
##            iterOpponent = random.randint(0,iterationLastSucces)
##            OpponentAgent = SelfplayAgent.SelfplayAgent(gc.STATECOUNT, gc.ACTIONS, gc.MODEL_FOLDER_PATH, iterOpponent) 
##            print("=========================================")
##            print("\n start training the agent against opponent, ", opp, "with brain: ", iterOpponent, "\n")
##            gm.TrainAgent(TheAgent, OpponentAgent, iteration)
#        # TODO remove these two lines: (zijn om tegen bot te runnen)
#        OpponentAgent = SimpleBot.SimpleBot(0)
#        gm.TrainAgent(TheAgent, OpponentAgent, renderScreen, iteration)
#        # After training multiple opponents, save the trained model
#        TheAgent.SaveBrainWeights(gc.MODEL_FOLDER_PATH, iteration)   
#
#        print("=========================================")
#        print("\n starting competitive games\n")
#        agentTotalScore = 0
#        oppTotalScore = 0
#        wins = 0
#        # This line will be put in the CSV file
#        toCSV = np.append([0, 0, iteration], np.zeros(gc.N_of_Test_Games + 3))
#        for tg in range(gc.N_of_Test_Games):
#            # Initiate the opponent random from the history
#            # TODO Only sample opponents from the ones that were succesful
#            # TODO this is swapped on when playing itself
##            iterOpponent = random.randint(0,iterationLastSucces)
##            toCSV[3 + tg] = iterOpponent
##            OpponentAgent = SelfplayAgent.SelfplayAgent(gc.STATECOUNT, gc.ACTIONS, gc.MODEL_FOLDER_PATH, iterOpponent) 
#            [scoreAI, scoreOld, hasWon] = gm.PlayGame(TheAgent, OpponentAgent, renderScreen, iteration)            
#            agentTotalScore += scoreAI
#            oppTotalScore += scoreOld
#            wins += hasWon
#        # Finish the toCSV line with the number of wins
#        toCSV[3 + gc.N_of_Test_Games] = wins
#        toCSV[3 + gc.N_of_Test_Games + 1] = agentTotalScore
#        toCSV[3 + gc.N_of_Test_Games + 2] = oppTotalScore
#        print(toCSV)
#        csvH.saveListToCSV(gc.CSV_FILE_PATH, toCSV)
#        
#        # if the new AI player is better then the old players, then use its brain next time
#        if (agentTotalScore > oppTotalScore):
#            iterationLastSucces = iteration
#            print("=========================================")
#            print("\n The new AI WON, Iteration last succes: ",iterationLastSucces, "\n")
#        else:
#            print("=========================================")
#            print("\n The new AI LOST at it: ", iteration, "\n")
#        print("The stats were: AgentTscore ", agentTotalScore, " OppTscore ", oppTotalScore, " wins ", wins, " out of ", gc.N_of_Test_Games)
#


if __name__ == "__main__":
    try:
        main()
        MyPong.quit_()
    except:
        print("Unexpected error")
        #Quit the pygame window
        MyPong.quit_()
        raise





