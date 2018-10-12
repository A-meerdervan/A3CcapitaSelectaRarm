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
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import load_model

#
#%% ==========================================================================
#  Keras based Nueral net Based Brain Class
class Brain:
    def __init__(self, NbrStates, NbrActions):
        self.NbrStates = NbrStates
        self.NbrActions = NbrActions
        self.type = -1 # This is used to help set paddle speed in MyPong.py
        #self.model = self._createModel()
        self.model = gc.getModel(gc.MODEL_ARCHITECTURE_TYPE)
        self.modelTarget = Sequential.from_config(self.model.get_config())
        #self.modelTarget = self.model
        print("MODEL TYPE! ", gc.MODEL_ARCHITECTURE_TYPE)
    

#    def _createModel(self):
#        
#        model = Sequential()
#
#        # Simple Model with Two Hidden Layers and a Linear Output Layer. The Input layer is simply the State input.
#        model.add(Dense(units=64, activation='relu', input_dim=self.NbrStates))
#        model.add(Dense(units=32, activation='relu'))
#        model.add(Dense(units=self.NbrActions, activation='linear'))				# Linear Output Layer as we are estimating a Function Q[S,A]
#        model.compile(loss='mse', optimizer='adam')     # use adam as an alternative optimsiuer as per comment
#
#        return model

#       model = Sequential()
#        # Simple Model with Two Hidden Layers and a Linear Output Layer. The Input layer is simply the State input.
#        model.add(Dense(units=20, activation='relu', input_dim=self.NbrStates))
#        model.add(Dense(units=20, activation='relu'))
#        model.add(Dropout(0.2))
#        model.add(Dense(units=20, activation='relu'))
#        model.add(Dropout(0.2))
#        model.add(Dense(units=20, activation='relu'))
#        model.add(Dropout(0.2))
#        model.add(Dense(units=20, activation='relu'))
#        model.add(Dropout(0.2))
#        model.add(Dense(units=self.NbrActions, activation='linear'))                # Linear Output Layer as we are estimating a Function Q[S,A]
#
#        model.compile(loss='mse', optimizer='adam')     # use adam as an alternative optimsiuer as per comment
#
#        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)
    
    def predictWithTargetQ(self, s):
        return self.modelTarget.predict(s)
    
    def updateQnetwork(self):
        self.modelTarget.set_weights(self.model.get_weights())

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.NbrStates)).flatten()

    def saveModel(self,filePath):
        self.model.save(filePath)
        
    def saveToArray(self):
        gc.MODELS = numpy.append(gc.MODELS, self.model)

    def loadBrain(self, iteration):
        self.model = gc.MODELS[iteration]

    def loadModel(self,filePath):
        self.model = load_model(filePath)


# =======================================================================================
# A simple Experience Replay memory
#  DQN Reinforcement learning performs best by taking a batch of training samples across a wide set of [S,A,R, S'] expereiences
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
    
# =====================================================================================
# See GlobalGameVariables.py for the rest of the parameters
# Speed of Epsilon decay
LAMBDA = -1*numpy.log(0.004) / (gc.TRAIN_TIME)
# ============================================================================================
class Agent:
    def __init__(self, NbrStates, NbrActions):
        self.NbrStates = NbrStates
        self.NbrActions = NbrActions
        self.brainNr = -1 # Changes when brain is loaded
        self.type = -1
        self.brain = Brain(NbrStates, NbrActions)
        self.ExpReplay = ExpReplay(gc.ExpReplay_CAPACITY)
        self.steps = 0
        self.epsilon = gc.MAX_EPSILON
        self.epsilonMature = gc.EPSILON_MATURE
    
    # ============================================
    # Reset the agent, to start learning again but not change its brain
    def reset(self):
        self.steps = 0
        self.ExpReplay = ExpReplay(gc.ExpReplay_CAPACITY)
        self.epsilon = gc.MAX_EPSILON

    # ============================================
    # Return the Best Action  from a Q[S,A] search.  Depending upon an Epslion Explore/ Exploitaiton decay ratio
    def Act_Train(self, s):
        if (random.random() < self.epsilon or self.steps < gc.OBSERVEPERIOD):
             # Explore
            return random.randint(0, self.NbrActions-1)                       
        else:
            # Exploit Brain best Prediction
            return numpy.argmax(self.brain.predictOne(s))                    
        
    def Act_Mature(self,s):
        if (random.random() < self.epsilonMature):
             # Explore
            return random.randint(0, self.NbrActions-1)                       
        else:
            # Exploit Brain best Prediction
            return numpy.argmax(self.brain.predictOne(s))                    

    # ============================================
    def CaptureSample(self, sample):  # in (s, a, r, s_) format
        self.ExpReplay.add(sample)

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        if(self.steps > gc.OBSERVEPERIOD):
            self.epsilon = gc.MIN_EPSILON + (gc.MAX_EPSILON - gc.MIN_EPSILON) * math.exp(-LAMBDA * (self.steps - gc.OBSERVEPERIOD))
            
        # Update the targetQ network every gc.TARGET_Q_NN_UPDATE_INTERVAL but not when it is 0
        if not (gc.TARGET_Q_NN_UPDATE_INTERVAL == 0):    
            if (self.steps % gc.TARGET_Q_NN_UPDATE_INTERVAL == 0) :
                self.brain.updateQnetwork()
                
        # TODO remove this, this is only for large runs
        if self.steps % gc.SAVE_MODEL_INTERVAL == 0:
            self.SaveBrainToFile(gc.MODEL_FOLDER_PATH, gc.MODEL_NR)
            gc.incrementModelNr()
            print("saved intermidiate model nr ",gc.MODEL_NR)

    def SaveBrain(self):
        self.brain.saveToArray()

    # change this to save all brains in the gc.MODELS
    def SaveBrainWeights(self, fromFile, modelsPath):
        for i in range(len(gc.MODELS)):
            filePath = modelsPath + "/model" + str(i) + ".hdf5"
            gc.MODELS[i].save(filePath)

    def SaveBrainToFile(self, modelsPath, iteration):
        filePath = modelsPath + "/model" + str(iteration) + ".hdf5"
        self.brain.saveModel(filePath)

    def LoadBrain(self, fromFile, modelsPath, iteration):
        #construct filepath
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
        # Reset relevant values back to 0 to be able to start training fresh
        self.reset()
        # The agent now knows what brain it has.
        self.brainNr = iteration

    # ============================================
    # Perform an Agent Training Cycle Update by processing a set of sampels from the Experience Replay memory
    def Process(self):
        batch = self.ExpReplay.sample(gc.BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.NbrStates)

        states = numpy.array([ batchitem[0] for batchitem in batch ])
        states_ = numpy.array([ (no_state if batchitem[3] is None else batchitem[3]) for batchitem in batch ])

        predictedQ = self.brain.predict(states)                        # Predict from keras Brain the current state Q Value
        predictedNextQ = self.brain.predictWithTargetQ(states_)                # Predict from keras Brain the next state Q Value

        x = numpy.zeros((batchLen, self.NbrStates))
        y = numpy.zeros((batchLen, self.NbrActions))

        #  Now compile the Mini Batch of [States, TargetQ] to Train an Target estimator of Q[S,A]
        for i in range(batchLen):
            batchitem = batch[i]
            state = batchitem[0]; a = batchitem[1]; reward = batchitem[2]; nextstate = batchitem[3]

            targetQ = predictedQ[i]
            if nextstate is None:
                targetQ[a] = reward                                                # An End state Q[S,A]assumption
            else:
                targetQ[a] = reward + gc.GAMMA * numpy.amax(predictedNextQ[i])       # The core Q[S,A] Update recursive formula

            x[i] = state
            y[i] = targetQ

        self.brain.train(x, y)                        #  Call keras DQN to Train against the Mini Batch set
# =======================================================================