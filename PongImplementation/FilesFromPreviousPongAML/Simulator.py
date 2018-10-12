# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:09:51 2018

@author: Alex
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 13:21:50 2018

@author: Alex
"""

#import MyPong # My PyGame Pong Game
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
        print("Used Folder ", modelsFolderPath, " used CSV ", csvFileName)
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
        self.theAgent.SaveBrainToFile(self.modelsPath, iteration)    
        self.theAgent.SaveBrain() # Save to the gc.Models array
            
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
        self.theAgent.SaveBrainToFile(self.modelsPath, iteration)   
        self.theAgent.SaveBrain() # Save to the gc.Models array
    
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
    
    # Save all brains from array to file in one go
    def saveAgentsToFile(self):
        self.theAgent.SaveBrainWeights(self.modelsPath)




