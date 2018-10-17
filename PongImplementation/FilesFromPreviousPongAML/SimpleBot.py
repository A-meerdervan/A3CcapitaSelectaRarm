# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:06:14 2018

@author: Alex
"""
from GlobalConstantsA3C import gc # has high level constants

class SimpleBot:
    def __init__(self, botType):
        # This determines the speed of its paddle
        self.type = botType
        # This is to show it does not have a brain
        self.brainNr = -1

    def Act_Mature(self, GameState):
        print(GameState)
        paddleYPos = GameState[0] * gc.NORMALISATION_FACTOR
        ballYPos = GameState[3] * gc.NORMALISATION_FACTOR
        #move up if ball is higher thn Openient Paddle
        # TODO zorgen dat paddle height enzo niet meer in pong.py staan
        if (paddleYPos + gc.PADDLE_HEIGHT/2 > ballYPos + gc.BALL_HEIGHT/2):
            return 1
        #move down if ball lower than Paddle
        if (paddleYPos + gc.PADDLE_HEIGHT/2 < ballYPos + gc.BALL_HEIGHT/2):
            return 2
        # not up or down, then do nothing = action = 0
        return 0
    