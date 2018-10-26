# -*- coding: utf-8 -*-
"""
Author: Arnold
Inspired by: https://github.com/dm-mch/DeepRL-Agents/blob/master/A3C-Doom.ipynb

This file is the same as the simulationA3C, exept that now the buffer is kept full,
new experiences are added and the oldest experience is dropped. This will help to
prevent the discounted reward problem. And the RNN is changed to a NN
"""

import threading
import multiprocessing
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import gobalConst as cn
#from skimage.color import rgb2gray # gave an error, had to pip it, then another error then had to pip scikit-image
#matplotlib inline
#from helper import * # had to pip it
#from vizdoom import *
#import gym # had to pip it and then had to pip gym[atari], then had to pip make, but better from right to left :P!
import SimulationEnvironment as sim
import pygame
#from random import choice
#from time import sleep
from time import time
import os
#import logging # logs messages in multithreading applications

#---- End imports ----

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_frame(frame):
    s = np.reshape(frame,[np.prod(frame.shape)])

    return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    rew = scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
#    print(x, len(x), rew)
    return rew

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            # s_size is input size
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)

            hidden = slim.fully_connected(self.inputs,512,activation_fn=tf.nn.elu)
            hidden2 = slim.fully_connected(hidden,512,activation_fn=tf.nn.elu)

            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(hidden2,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(hidden2,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)

            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 10e-6))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 10e-6)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.1
                self.adv_sum = tf.reduce_sum(self.advantages)

                #Get gradients from local network using local losses
                # uses tensorflow graphs to store the data of the session
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                # ???limits the values of the grads???
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40)

                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                # apply gradients using apply_gradients
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

class Worker():
    def __init__(self,name,s_size,a_size,trainer,model_path,tfSummary_path,global_episodes,global_rewardEndEpisode):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.global_rewardEndEpisode = global_rewardEndEpisode
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(tfSummary_path[2:] + str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)

        self.actions = [1,2,3,4,5,6]
        self.sleep_time = 0.028
        self.env = sim.SimulationEnvironment()
        self.r = 0

    def train(self,global_AC,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
#        next_observations = rollout[:,3]
        values = rollout[:,5]

        if(len(rewards) < 30):
#            rewards = np.asarray(rewards)
            x = np.repeat([rollout[0]], 30-len(rewards), axis=0)
            rollout = np.concatenate((rollout, x), axis=0)

            rollout = np.array(rollout)
            observations = rollout[:,0]
            actions = rollout[:,1]
            rewards = rollout[:,2]
#            next_observations = rollout[:,3]
            values = rollout[:,5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
#        print('ee ',  self.rewards_plus)
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]   #the :-1 takes only the first decimal
        """Trains when the episode has ended, but then the rewards array is shorter, thus the discounted
        cumelative reward is much lower. The agent cannot train when it has terminated, or its reward needs to be extended
        with the last received reward untill lenth is equal. This is implemented this way because it does work with sparse
        rewards, then you dont have such problems"""

        #TODO: why only use the first decimal?
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
#        rnn_state = self.local_AC.state_init
        # feed the placeholder of the AC_NETWORK
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages
            }
        v_l,p_l,e_l,g_n,v_n,adv, apl_g = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.adv_sum,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n, v_n, adv/len(rollout)

    def work(self,max_episode_length,gamma,global_AC,sess,coord,saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        steps_train = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                s = self.env.reset()
                episode_frames.append(s)
                s = process_frame(s)

                done = False
                while not done:
                    #Take an action using probabilities from policy network output.
                    a_dist,v = sess.run([self.local_AC.policy,self.local_AC.value],
                        feed_dict={self.local_AC.inputs:[s]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
#                    print(a_dist)
                    a = np.argmax(a_dist == a)

                    s1,r,d = self.env.step(self.actions[a])

                    episode_frames.append(s1)
                    s1 = process_frame(s1)

                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == cn.run_BufferSize and d != True:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value,
                            feed_dict={self.local_AC.inputs:[s]})[0,0]
                        v_l,p_l,e_l,g_n,v_n, adv = self.train(global_AC,episode_buffer,sess,gamma,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if  episode_step_count >= max_episode_length - 1 or d == True:
                        self.r = r
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                #Alex added: Store a global end of episode reward to print it
                sess.run(self.global_rewardEndEpisode.assign(int(episode_reward)))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n,adv = self.train(global_AC,episode_buffer,sess,gamma,0.0)


                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 20 == 0 and episode_count != 0:
                    if episode_count % 500 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Perf/Average reward', simple_value=float(episode_reward/episode_step_count))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Advantage', simple_value=float(adv))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    summary.value.add(tag='Terminal Reward', simple_value=float(self.r*200))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                #if self.name == 'worker_0':
                sess.run(self.increment)
                episode_count += 1


#max_episode_length = 10000
#gamma = .99 # discount rate for advantage estimation and reward discounting
#s_size = 17 # Observations are greyscale frames of 84 * 84 * 1
#a_size = 6 # Agent can rotate each joint in 2 directions
#load_model = False
#model_path = './model'
#frameRate = 15
#showScreen = True

tf.reset_default_graph()

if not os.path.exists(cn.run_ModelPath):
    os.makedirs(cn.run_ModelPath)

#Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')


global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
global_rewardEndEpisode = tf.Variable(0,dtype=tf.int32,name='global_rewardEndEpisode',trainable=False)
trainer = tf.train.RMSPropOptimizer(learning_rate=cn.run_LearningRate, decay=0.99, epsilon=0.1)
master_network = AC_Network(cn.run_sSize,cn.run_aSize,'global',None) # Generate global network
num_workers = 4 #num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
workers = []
# Create worker classes
for i in range(num_workers):
    workers.append(Worker(i,cn.run_sSize,cn.run_aSize,trainer,cn.run_ModelPath,cn.TF_SUMM_PATH,global_episodes,global_rewardEndEpisode))
saver = tf.train.Saver(max_to_keep=5)

# Run everything in a tensforflow session called sess
with tf.Session() as sess:
    coord = tf.train.Coordinator()  # coordinates communication between threads
    # A: I think this means that it is able to continiue where it left off in a previous run
    # If load_model == False, then you just start from scratch
    if cn.run_LoadModel == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(cn.run_ModelPath)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    # this list contains the worker threads which are running
       # solves the problem of making it responsive
    workers[0].work(cn.run_MaxEpisodeLenght,cn.run_Gamma,master_network,sess,coord,saver)
#
#    worker_threads = []
##     Loop the workers and start a thread for each
#    for worker in workers:
#        # define a function that executes the work method of each worker
#        worker_work = lambda: worker.work(cn.run_MaxEpisodeLenght ,cn.run_Gamma,master_network,sess,coord,saver)
#        # execute the function in a new thread
#        t = threading.Thread(target=(worker_work))
#        # start thread
#        t.start()
#        # keep track of the threads
#        worker_threads.append(t)
    gs = 0
#     The coordinator has the overview and while it wants to continue:
#     This routine probes the training proces and prints an update, every 10 seconds.
#    if showScreen:
#        workers[0].env.initScreen()

    clock = pygame.time.Clock()
    ctr = 0
    while not coord.should_stop():
        clock.tick(cn.run_FPS)
        # global episodes is a global tf variable which is synced over all threads
        # Now the current episode nr is obtained:
        if (ctr / cn.run_FPS == 10):
            s = time()
            gs1 = sess.run(global_episodes)  # get value of variable
            lastRewardEndOfEp = sess.run(global_rewardEndEpisode)
            print("Episodes", gs1, 'one for ', (time()-s)/(gs1-gs), "\n" + "Reward at end of episode: ",lastRewardEndOfEp)
            gs = gs1
            ctr = 0

        if cn.run_Render:

            workers[0].env.render()
            pygame.event.get()

        ctr += 1

    coord.join(worker_threads)   # terminate the threads


