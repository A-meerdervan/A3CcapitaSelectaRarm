# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:16:03 2018

@author: Alex
"""
import tensorflow as tf
import numpy as np
import os# Alex: I added this
import multiprocessing
# Own files
from AC_Network import AC_Network
from Worker import Worker
from Env.GlobalConstantsA3C import gc # has high level constants
import Env.A3CenvPong # import custom Pong env (no images but 6 numbers as state)

# end imports

class EvalWorker():
    def __init__(self,actionSpace,network,sess,save,verbose,max_episode_length):
        self.local_AC = network
        self.sess = sess
        self.save = save
        self.max_episode_length = max_episode_length
        self.env = Env.A3CenvPong.A3CenvPong()
        self.actions = actionSpace
        self.verbose = verbose
        
    def play1Game(self):
        # stuff to keep track of
        episode_values = []
        episode_reward = 0
        episode_step_count = 0

        # Start the game
        s = self.env.reset()
        done = False
        while ( not done):
            if gc.RENDER_SCREEN: self.env.render()
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
        # Specific to our pong implementation
        self.env.quitScreen()
        results[n,:] = [np.mean(results[:n,0]),np.mean(results[:n,1]),np.mean(results[:n,2])]
        results[n+1,:] = [np.std(results[:n,0]),np.std(results[:n,1]),np.std(results[:n,2])]
        if self.verbose: 
            print(results[n,:],"the means", )
            print(results[n + 1,:],"the standard devs", )
            print("ewa ik ben klaar met evalueren man")
        
    
        

max_episode_length = 10000
s_size = 6 # Our pong version has a state of 6 numbers
a_size = 3 # Agent can move up down or do nothing
load_model = True
model_path = './model'

tf.reset_default_graph()

# TODO: use this to store some results
#Create a directory to save episode playback gifs to 
if not os.path.exists('./results'):
    os.makedirs('./results')

global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
# Alex added: This is used to print the current reward at the end of an episode
global_rewardEndEpisode = tf.Variable(0,dtype=tf.int32,name='global_rewardEndEpisode',trainable=False)

trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=0.99, epsilon=0.1)
master_network = AC_Network(s_size,a_size,'global',None) # Generate global network
num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
workers = []
# Create worker classes
for i in range(num_workers):
    workers.append(Worker(i,s_size,a_size,trainer,model_path,global_episodes,global_rewardEndEpisode))
saver = tf.train.Saver(max_to_keep=5)

# A: I think this means that it is able to continiue where it left off in a previous run
# If load_model == False, then you just start from scratch
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
     
   # Start 1 EvalWorker
    save = False # not used yet, but should save things to file or something
    verbose = True # if true it prints results, if not it prints nothing
    # pong specific action space
    actionSpace = [1,2,3]
    evalWorker = EvalWorker(actionSpace,master_network,sess,save,verbose,max_episode_length)
    evalWorker.playNgames(1)



