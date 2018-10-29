# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:21:46 2018

@author: Alex
"""
import Env.MyPong as MyPong
import Env.SimpleBot as SimpleBot # needed for the opponent in the env.
#from Env.GlobalConstantsA3C import gc # has high level constants

# =======================================================================

class A3CenvPong:
    # init with draw to tell whether to render a screen
    def __init__(self,draw):
        self.botType = 10 # this is the speed of the bot paddle 10 is standard
        # TODO miss de draw uit de init van PongGame halen
        self.theGame = MyPong.PongGame(self.botType, draw)
        self.oppAgent = SimpleBot.SimpleBot(self.botType) 
        self.screenActive = False
        
    def reset(self):
        # resets the env and returns the initial state
        return self.theGame.resetEpisode()
        
    def render(self):
        if (not self.screenActive):
            self.theGame.createWindow()
            self.screenActive = True
        self.theGame.renderCurrentState()
    # This function takes the action chosen by the agent and moves the environment
    # 1 timestep to the future and returns the new reward, state and such
    def step(self, actionNr):
        # Determine next action from opponent
        gameStateOpp = self.theGame.returnCurrentStateOpp()
        oppAction = self.oppAgent.Act_Mature(gameStateOpp)
        # This function returns the new state and the obtained reward
        s1,r,d,i = self.theGame.PlayNextMoveA3C(actionNr, oppAction)
        # Return: The next state, the imidiate reward, done = episode terminated?, i = the timestep
        return s1,r,d,i
    def quitScreen(self):
        self.theGame.quit_()
        


    