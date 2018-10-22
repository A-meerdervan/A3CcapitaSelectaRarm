# -*- coding: utf-8 -*-
"""
Author: Alex
Inspired by: https://github.com/dm-mch/DeepRL-Agents/blob/master/A3C-Doom.ipynb
"""

import threading
import multiprocessing
#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow.contrib.slim as slim
#import scipy.signal
#from skimage.color import rgb2gray # gave an error, had to pip it, then another error then had to pip scikit-image
#import skimage
#matplotlib inline
#from helper import * # had to pip it
#from vizdoom import *
#import gym # had to pip it and then had to pip gym[atari], then had to pip make, but better from right to left :P!

#from random import choice
from time import sleep
from time import time
import os# Alex: I added this

# Alex: added this
#from Env.GlobalConstantsA3C import gc # has high level constants
from Worker import Worker # the worker which acts and trains its brain
from AC_Network import AC_Network # The network used as brain


#---- End imports ----

# Run specific parameters
max_episode_length = 10000
# TODO: Was eerst .99 maar vanwege onze paper overgezet naar .96
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = 6 # Our pong version has a state of 6 numbers
a_size = 3 # Agent can move up down or do nothing
learningRate = 1e-3 # was at 7e-4 for the Dm-mc implementation
load_model = False
outputs_folder = '/testRUN3'
num_workers = 16 #multiprocessing.cpu_count() # Set workers ot number of available CPU threads

# Not run specific parameters
outputs_folder = './LogsOfRuns' + outputs_folder
tfSummary_path = outputs_folder + '/train_'
model_path = outputs_folder + '/model'
#frames_path = outputs_folder + '/frames'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
##Create a directory to save episode playback gifs to
#if not os.path.exists(frames_path):
#    os.makedirs(frames_path)

global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
# Alex added: This is used to print the current reward at the end of an episode
global_rewardEndEpisode = tf.Variable(0,dtype=tf.int32,name='global_rewardEndEpisode',trainable=False)

trainer = tf.train.RMSPropOptimizer(learning_rate=learningRate, decay=0.99, epsilon=0.1)
master_network = AC_Network(s_size,a_size,'global',None) # Generate global network

workers = []
# Create worker classes
for i in range(num_workers):
    workers.append(Worker(i,s_size,a_size,trainer,model_path,tfSummary_path,global_episodes,global_rewardEndEpisode))
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
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    # this list contains the worker threads which are running
    worker_threads = []
   # Loop the workers and start a thread for each
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,master_network,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        worker_threads.append(t)
    gs = 0
    # The coordinator has the overview and while it wants to continue:
    # This routine probes the training proces and prints an update, every 10 seconds.
    while not coord.should_stop():
        s = time()
        sleep(10)
        # global episodes is a global tf variable which is synced over all threads
        # Now the current episode nr is obtained:
        gs1 = sess.run(global_episodes)
        lastRewardEndOfEp = sess.run(global_rewardEndEpisode)
        print("Episodes", gs1, 'one for ', (time()-s)/(gs1-gs), "\n" + "Reward at end of episode: ",lastRewardEndOfEp)
        
        gs = gs1
    coord.join(worker_threads)





