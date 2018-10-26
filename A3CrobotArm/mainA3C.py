# -*- coding: utf-8 -*-
"""
Author: Alex
Inspired by: https://github.com/dm-mch/DeepRL-Agents/blob/master/A3C-Doom.ipynb
"""

import threading
import tensorflow as tf

#from random import choice
from time import sleep
from time import time
import os# Alex: I added this

# Alex: added this
#from Env.GlobalConstantsA3C import gc # has high level constants

from Worker import Worker # the worker which acts and trains its brain
from AC_Network import AC_Network # The network used as brain
#from runConfig import rc # has high level constants
import runConfig as rc # has high level constants

# Added to be able to run the our robot arm sim
import pygame # for the clock to wait with showing a render (not every frame is renderd)
#import logging # logs messages in multithreading applications


#---- End imports ----

tf.reset_default_graph()

if not os.path.exists(rc.MODEL_PATH):
    os.makedirs(rc.MODEL_PATH)

# This only tracks the episode count of worker0 but it is globally availlable to be able to print in the main thread
global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
# Alex added: This is used to print the current reward at the end of an episode (updated by all workers)
global_rewardEndEpisode = tf.Variable(0,dtype=tf.float32,name='global_rewardEndEpisode',trainable=False)

trainer = tf.train.RMSPropOptimizer(learning_rate=rc.L_RATE, decay=rc.ALPHA, epsilon=rc.EPSILON)
master_network = AC_Network(rc.S_SIZE,rc.A_SIZE,'global',None) # Generate global network

workers = []
# Create worker classes which will later work in parallel
for i in range(rc.NUM_WORKERS):
    workers.append(Worker(i,rc.S_SIZE,rc.A_SIZE,trainer,rc.MODEL_PATH,rc.TF_SUMM_PATH,global_episodes,global_rewardEndEpisode))
saver = tf.train.Saver(max_to_keep=5)

# Run everything in a tensforflow session called sess
# A: I think this means that it is able to continiue where it left off in a previous run
# If load_model == False, then you just start from scratch
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if rc.LOAD_MODEL == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(rc.MODEL_PATH)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    # this list contains the worker threads which are running
    worker_threads = []
   # Loop the workers and start a thread for each
    for worker in workers:
        # define a function that executes the work method of each worker
        worker_work = lambda: worker.work(rc.MAX_EP_LENGTH,rc.GAMMA,master_network,sess,coord,saver)
        # execute the function in a new thread
        t = threading.Thread(target=(worker_work))
        # start thread
        t.start()
        # keep track of the threads
        worker_threads.append(t)
    gs = 0
    # The coordinator has the overview and while it wants to continue:
    # This routine probes the training proces and prints an update, every 10 seconds.
    # This is used to give a visual update in case of the robot arm
    clock = pygame.time.Clock()
    ctr = 0 # keeps track of at which frame you are at
    while not coord.should_stop():
        clock.tick(rc.FPS)
        # global episodes is a global tf variable which is synced over all threads
        # This if, prints the last known total reward of an episode every 10s
        if (ctr / rc.FPS == 10):
            s = time()
            gs1 = sess.run(global_episodes)  # get value of variable
            lastRewardEndOfEp = sess.run(global_rewardEndEpisode)
            print("Episodes", gs1, 'one for ', (time()-s)/(gs1-gs), "\n" + "Reward at end of episode: ",lastRewardEndOfEp)
            gs = gs1
            ctr = 0
        # This if, renders the env at rc.FPS times per second, but it does not slow down the learning process
        if rc.RENDER_SCREEN:
            workers[0].env.render()
            pygame.event.get()
        ctr += 1 # this counter reaches a max of 10*rc.FPS and is then set to 0 again
    coord.join(worker_threads)





