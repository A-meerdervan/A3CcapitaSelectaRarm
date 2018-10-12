# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:59:23 2018

@author: arnold
"""

#
# This Agent demonstrates use of a Keras centred Q-network estimating the Q[S,A] Function from a few basic Features
#
# This DQN Agent Software is Based upon the following  Jaromir Janisch  source:
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# as employed against OpenAI  Gym Cart Pole examples
#  requires keras [and hence Tensorflow or Theono backend]
# ==============================================================================
import random, numpy, math
from GlobalConstants import gc
#
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model
#
#%% ==========================================================================
#  Keras based Nueral fnet Based Brain Class

class Brain:
    def __init__(self, NbrStates, NbrActions):
        self.NbrStates = NbrStates
        self.NbrActions = NbrActions

        # Always create the model, eventhough it will be overwritten by a load or save
        #self.model = self._createModel()
        self.model = gc.getModel(gc.MODEL_ARCHITECTURE_TYPE)

#    def _createModel(self):
#        model = Sequential()
#
#        # Simple Model with Two Hidden Layers and a Linear Output Layer. The Input layer is simply the State input.
#        model.add(Dense(units=64, activation='relu', input_dim=self.NbrStates))
#        model.add(Dense(units=32, activation='relu'))
#        model.add(Dense(units=32, activation='relu'))
#        model.add(Dense(units=self.NbrActions, activation='linear'))                # Linear Output Layer as we are estimating a Function Q[S,A]
#
#        model.compile(loss='mse', optimizer='adam')     # use adam as an alternative optimsiuer as per comment
#
#        return model

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.NbrStates)).flatten()

    def saveModel(self,filePath):
        self.model.save(filePath)

    def loadModel(self,filePath):
        self.model = load_model(filePath)
        
    def loadBrain(self, iteration):
        self.model = gc.MODELS[iteration]


# =======================================================================================
# A simple Experience Replay memory
#  DQN Reinforcement learning performs best by taking a batch of training
#  samples across a wide set of [S,A,R, S'] expereiences
#
class ExpReplay:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

# ============================================================================================
class SelfplayAgent:
    def __init__(self, NbrStates, NbrActions, fromFile, modelsPath, iteration):
        self.NbrStates = NbrStates
        self.NbrActions = NbrActions
        self.epsilonMature = gc.EPSILON_MATURE
        self.type = -1 # This is used to help set paddle speed in MyPong.py
        self.steps = 0
        # The agent now knows what brain it has.
        self.brainNr = iteration 

        self.brain = Brain(NbrStates, NbrActions)
        
        filePath = ""
        # Only load the initial brain at 0, if starting from scratch
        if (iteration == 1 and not gc.startFromScratch):
            filePath =  modelsPath + "/" + gc.INITIAL_MODEL_NAME + ".hdf5"
            self.brain.loadModel(filePath)
        elif (iteration > 1):
            if fromFile: 
                filePath =  modelsPath + "/" + "model" + str(iteration -1) + ".hdf5"
                self.brain.loadModel(filePath)
            else:
                self.brain.loadBrain(iteration - 1)

#    def Act(self, st):
#        #change the state x speeds
#        st[2] = 1-st[2]   #change bal x
#        st[4] = -1*st[4]  #change bal x direction
#        return numpy.argmax(self.brain.predictOne(st))  # Exploit Brain best Prediction
    
    # ============================================
    # Return the Best Action  from a Q[S,A] search.
    # NOT training mode, but competitive mature mode
    def Act_Mature(self,s):
        #change the state x speeds
        # Change the ball's x position. By subtracting the x from the max x
        # Where the max x, is the screenwidth divided by the Normalization factor
        s[2] = (gc.WINDOW_WIDTH/gc.NORMALISATION_FACTOR) - s[2]
        s[4] = -1*s[4]  #change bal x direction
        
        if (random.random() < self.epsilonMature):
             # Explore
            return random.randint(0, self.NbrActions-1)                       
        else:
            # Exploit Brain best Prediction
            return numpy.argmax(self.brain.predictOne(s))   
