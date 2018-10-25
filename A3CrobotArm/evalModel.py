# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:16:03 2018

@author: Alex
"""
import tensorflow as tf
import numpy as np
import os# Alex: I added this
#import multiprocessing
import pygame
# Own files
from AC_Network import AC_Network
from Worker import Worker
#from Env.GlobalConstantsA3C import gc # has high level constants
#from runConfig import rc # has high level constants
import runConfig as rc # has high level constants

import Env.A3CenvPong # import custom Pong env (no images but 6 numbers as state)
import SimulationEnvironment as sim # own Rarm sim


# end imports

class EvalWorker():
    def __init__(self,envIsRarm,actionSpace,network,sess,save,verbose,max_episode_length):
        self.local_AC = network
        self.sess = sess
        self.save = save
        self.max_episode_length = max_episode_length
        self.env = self.getEnv(rc.ENV_IS_RARM)
        self.actions = actionSpace
        self.verbose = verbose
        self.clock = pygame.time.Clock()
        
    def play1Game(self):
        # stuff to keep track of
        episode_values = []
        episode_reward = 0
        episode_step_count = 0

        # Start the game
        s = self.env.reset()
        done = False
        while ( not done):
            # When render, always do env.render(), but slow down if needed
            if rc.RENDER_SCREEN: 
                if rc.SHOW_NORMAL_SPEED:
                    self.clock.tick(rc.FPS)
                self.env.render()
            #Take an action using probabilities from policy network output.
            a_dist,v = sess.run([self.local_AC.policy,self.local_AC.value],
                feed_dict={self.local_AC.inputs:[s]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)

            # Take one step in env using the chosen action a
            s1,r,d, i = self.env.step(self.actions[a])
    
            s = s1 # needed for next timestep    
            # bookkeeping
            episode_reward += r
            episode_step_count += 1
            episode_values.append(v[0,0])
                    
            # If the episode terminates or the max steps has been reached
            # Then the episode (and so this while loop) terminates.
            if  episode_step_count >= self.max_episode_length - 1 or d == True:
                break
        # Now that the episode has terminated, store the relevant
        # statistics                            
        episode_mean_value = np.mean(episode_values)
        return [episode_reward,episode_mean_value,episode_step_count]
    
    def playNgames(self,n):
        # The last 2 rows are for the averages and the standard devs
        results = np.zeros([n + 2 , 3]) # watchout the 3 is hardcoded
        for i in range(n):
            # This returns the results from the match
            results[i,:] = self.play1Game()
            if self.verbose: print("eval game ", i, " sumR,meanV,tsteps ", results[i,:])

        results[n,:] = [np.mean(results[:n,0]),np.mean(results[:n,1]),np.mean(results[:n,2])]
        results[n+1,:] = [np.std(results[:n,0]),np.std(results[:n,1]),np.std(results[:n,2])]
        # Specific to our pong implementation
        if not rc.ENV_IS_RARM: self.env.quitScreen()
        if self.verbose: 
            print(results[n,:],"the means", )
            print(results[n + 1,:],"the standard devs", )
            print("ewa ik ben klaar met evalueren man")
    
    # Get the right env
    def getEnv(self,envIsRarm):
        if envIsRarm:
            return sim.SimulationEnvironment(rc.sim_RandomGoal)    
        # Play Pong S6
        else:
            return Env.A3CenvPong.A3CenvPong(rc.RENDER_SCREEN)
        
        
# Here are the parameters that should match the run which is being evaluated
# --------------------------------------
cpu_count_training = 2 # Number of cpu's used during training
evalFolder = 'RarmVerwijderDit' # The folder that holds the model and train folders
nrOfEvalGames = 10
            
max_episode_length = 10000
load_model = True # must be True to evaluate a given model file
model_path = './LogsOfRuns/' + evalFolder + "/model"
# These should be added for better generality:
results_path = './results'
# Maybe the learning rate enzo, maar miss maakt dat niet uit. (dingen die de trainer meekrijgt)
# Iets over de AC_Network dat gebruikt word. Anders gaat het stuk

# Here are parameters which can be set independent of what training session it had
# ---------------------------------------
save = False # not used yet, but should save things to file or something
verbose = True # if true it prints results, if not it prints nothing

tf.reset_default_graph()

# TODO: use this to store some results
#Create a directory to save episode playback gifs to 
if not os.path.exists(results_path):
    os.makedirs(results_path)

global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
# Alex added: This is used to print the current reward at the end of an episode
global_rewardEndEpisode = tf.Variable(0,dtype=tf.int32,name='global_rewardEndEpisode',trainable=False)

trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=0.99, epsilon=0.1)
master_network = AC_Network(rc.S_SIZE,rc.A_SIZE,'global',None) # Generate global network
num_workers = cpu_count_training
workers = []
# Create worker classes
for i in range(num_workers):
    workers.append(Worker(i,rc.S_SIZE,rc.A_SIZE,trainer,model_path,rc.TF_SUMM_PATH,global_episodes,global_rewardEndEpisode))
saver = tf.train.Saver(max_to_keep=5)

# A: I think this means that it is able to continiue where it left off in a previous run
# If load_model == False, then you just start from scratch
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
   # Start 1 EvalWorker
    evalWorker = EvalWorker(rc.ENV_IS_RARM,rc.ACTION_SPACE,master_network,sess,save,verbose,max_episode_length)
    evalWorker.playNgames(nrOfEvalGames)



