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
from runConfig import rc # has high level constants

#---- End imports ----

tf.reset_default_graph()

if not os.path.exists(rc.MODEL_PATH):
    os.makedirs(rc.MODEL_PATH)
    
##Create a directory to save episode playback gifs to
#if not os.path.exists(frames_path):
#    os.makedirs(frames_path)

global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
# Alex added: This is used to print the current reward at the end of an episode
global_rewardEndEpisode = tf.Variable(0,dtype=tf.int32,name='global_rewardEndEpisode',trainable=False)

trainer = tf.train.RMSPropOptimizer(learning_rate=rc.L_RATE, decay=rc.ALPHA, epsilon=rc.EPSILON)
master_network = AC_Network(rc.S_SIZE,rc.A_SIZE,'global',None) # Generate global network

workers = []
# Create worker classes
for i in range(rc.NUM_WORKERS):
    workers.append(Worker(i,rc.S_SIZE,rc.A_SIZE,trainer,rc.MODEL_PATH,rc.TF_SUMM_PATH,global_episodes,global_rewardEndEpisode))
saver = tf.train.Saver(max_to_keep=5)

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
        worker_work = lambda: worker.work(rc.MAX_EP_LENGTH,rc.GAMMA,master_network,sess,coord,saver)
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





