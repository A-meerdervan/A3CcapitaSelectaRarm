# -*- coding: utf-8 -*-
"""
Author: Arnold
Inspired by: https://github.com/dm-mch/DeepRL-Agents/blob/master/A3C-Doom.ipynb

This file is the same as the simulationA3C, exept that now the buffer is kept full,
new experiences are added and the oldest experience is dropped. This will help to
prevent the discounted reward problem. And the RNN is changed to a NN
"""

import threading
#import multiprocessingq
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
import Env.A3CenvPong # import custom Pong env (no images but 6 numbers as state)
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

            hidden = slim.fully_connected(self.inputs,cn.netw_nHidNodes,activation_fn=tf.nn.elu)
            hidden2 = slim.fully_connected(hidden,cn.netw_nHidNodes,activation_fn=tf.nn.elu)

            # -- Specifies a LSTM cell with all its vars -- #
            #Recurrent network for temporal dependencies
#            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(cn.netw_nHidNodes,state_is_tuple=True)
#            # init hidden state
#            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
#            # init output state
#            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
#            self.state_init = [c_init, h_init]
#            # placeholder: way to feed data into a tensor
#            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
#            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
#            self.state_in = (c_in, h_in)
#            rnn_in = tf.expand_dims(hidden2, [0])
#            step_size = tf.shape(hidden2)[:1]
#            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
#            # define the lstm cells
#            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
#                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
#                time_major=False)
#            lstm_c, lstm_h = lstm_state
#            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
#            rnn_out = tf.reshape(lstm_outputs, [-1, cn.netw_nHidNodes])

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
                self.loss = cn.netw_vfCoef * self.value_loss + self.policy_loss - self.entropy * cn.netw_entCoef
                self.adv_sum = tf.reduce_sum(self.advantages)

                #Get gradients from local network using local losses
                # uses tensorflow graphs to store the data of the session
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                # ???limits the values of the grads???
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,cn.netw_maxGradNorm)

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
        self.episode_wallHitPercentages = []
        self.episode_terminalRewards = []
        self.summary_writer = tf.summary.FileWriter(tfSummary_path[2:] + str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)

        self.actions = cn.run_actionSpace
        self.env = sim.SimulationEnvironment()

    def train(self,global_AC,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
#        next_observations = rollout[:,3]
        values = rollout[:,5]

        """
        This was Arnolds patch to deal with termination of the episode when hitting a wall
        Now it is not needed anymore and actually causes instability and a too low estimate of the value
        of reaching the goal.
        Trains when the episode has ended, but then the rewards array is shorter, thus the discounted
        cumelative reward is much lower. The agent cannot train when it has terminated, or its reward needs to be extended
        with the last received reward untill lenth is equal. This is implemented this way because it does work with sparse
        rewards, then you dont have such problems"""
#        if(len(rewards) < cn.run_BufferSize):
##            rewards = np.asarray(rewards)
#            x = np.repeat([rollout[0]], cn.run_BufferSize-len(rewards), axis=0)
#            rollout = np.concatenate((rollout, x), axis=0)
#
#            rollout = np.array(rollout)
#            observations = rollout[:,0]
#            actions = rollout[:,1]
#            rewards = rollout[:,2]
##            next_observations = rollout[:,3]
#            values = rollout[:,5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]   #the :-1 takes only the first decimal
        # the advantages are calculated by taking the discounted rewards and subtracting the baseline, the value.
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = discounted_rewards - self.value_plus[:-1]
        # This uses generalised advantage estimation. But we have not used this:
#        tdErrors = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
#        advantages = discount(tdErrors,Lambda*gamma)
        
        

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
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0.0
                d = False

                s = self.env.reset()

                while True:
                    #Take an action using probabilities from policy network output.
                    a_dist,v = sess.run([self.local_AC.policy,self.local_AC.value],
                        feed_dict={self.local_AC.inputs:[s]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)

#                    s1,r,d,epLength = self.env.step(self.actions[a])
                    s1,r,d,epLength = self.env.step(self.actions[a])

                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1

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
                    if  epLength >= max_episode_length - 1 or d == True:
                        break
                # here the episode has endend, now do bookkeeping:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(epLength)
                self.episode_mean_values.append(np.mean(episode_values))
                if cn.ENV_IS_RARM: self.episode_wallHitPercentages.append(self.env.wallHits/epLength)
                self.episode_terminalRewards.append(r)
                # print terminal reward
#                print('terminalR ',r)

                #Alex added: Store a global end of episode reward to print it
                sess.run(self.global_rewardEndEpisode.assign(int(episode_reward)))

                # Update the network using the experience buffer at the end of the episode.
                # But only update if the termination was not caused by reaching max steps
                if len(episode_buffer) != 0 and epLength < max_episode_length:
                    v_l,p_l,e_l,g_n,v_n,adv = self.train(global_AC,episode_buffer,sess,gamma,0.0)


                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % cn.run_TFsummIntrvl == 0 and episode_count != 0:
                    if episode_count % cn.run_TFmodelSaveIntrvl == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-cn.run_TFsummIntrvl:])
                    mean_length = np.mean(self.episode_lengths[-cn.run_TFsummIntrvl:])
                    mean_value = np.mean(self.episode_mean_values[-cn.run_TFsummIntrvl:])
                    mean_terminalRs = np.mean(self.episode_terminalRewards[-cn.run_TFsummIntrvl:])
                    summary = tf.Summary()
                    # Add the percentage of timesteps that the agent wanted to hit the wall
                    if cn.ENV_IS_RARM:
                        mean_WallHitPerctg = np.mean(self.episode_wallHitPercentages[-cn.run_TFsummIntrvl:])
                        summary.value.add(tag='Perf/WallHitPercentage'
                                                         , simple_value=float(mean_WallHitPerctg))
                    summary.value.add(tag='Perf/Terminal Reward', simple_value=float(mean_terminalRs * cn.sim_rewardNormalisation))
                    summary.value.add(tag='Perf/SumReward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/AvrgValue', simple_value=float(mean_value))
                    summary.value.add(tag='Perf/Average reward', simple_value=float(episode_reward/i))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Advantage', simple_value=float(adv))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                # only print in main the nr of episodes of worker_0
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

class EvalWorker():
    def __init__(self,envIsRarm,actionSpace,network,sess,save,verbose,max_episode_length):
        self.local_AC = network
        self.sess = sess
        self.save = save
        self.max_episode_length = max_episode_length
        self.env = self.getEnv(cn.ENV_IS_RARM)
        self.actions = actionSpace
        self.verbose = verbose
        self.clock = pygame.time.Clock()
        self.render = cn.EVAL_RENDER

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
            if self.render:
                if cn.EVAL_SHOW_NORMAL_SPEED:
                    self.clock.tick(cn.EVAL_FPS)
                self.env.render()
                pygame.event.get()
            #Take an action using probabilities from policy network output.
            a_dist,v = sess.run([self.local_AC.policy,self.local_AC.value],
                feed_dict={self.local_AC.inputs:[s]})
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)

            # Take one step in env using the chosen action a
#
            if (cn.REAL_SETUP):
                s1,r,d, i = self.env.stepRealWorld(self.actions[a])
            else:
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
        won = 0.
        if r * cn.sim_rewardNormalisation == cn.sim_GoalReward:
            won = 1.
        #episode_mean_value = np.mean(episode_values)
        wallHitFactor = self.env.wallHits/episode_step_count
        return [episode_reward,episode_step_count,wallHitFactor,won]

    def playNgames(self,n):
        # The last 2 rows are for the averages and the standard devs
        results = np.zeros([n + 2 , 4]) # watchout the 4 is hardcoded
        for i in range(n):
            # This returns the results from the match
            results[i,:] = self.play1Game()
            if self.verbose: print("eval game ", i, " sumR,Length,wallHitF.,Won?", results[i,:])

        results[n,:] = [np.mean(results[:n,0]),np.mean(results[:n,1]),np.mean(results[:n,2]),np.mean(results[:n,3])]
        results[n+1,:] = [np.std(results[:n,0]),np.std(results[:n,1]),np.std(results[:n,2]),np.std(results[:n,3])]

        # release video
        if cn.REAL_SETUP: self.env.markerDetector.releaseVideo()
        # Specific to our pong implementation
        if not cn.ENV_IS_RARM: self.env.quitScreen()
        if self.verbose:
            print('sumR,Length,wallHitF.,Won?')
            print(results[n,:],"the means", )
            print(results[n + 1,:],"the standard devs", )
            print("ewa ik ben klaar met evalueren man")

    # Get the right env (nog niet aangepast voor pong env. de global const mist allemaal dingen)
    def getEnv(self,envIsRarm):
        if envIsRarm:
            return sim.SimulationEnvironment(cn.REAL_SETUP)
        # Play Pong S6
        else:
            return Env.A3CenvPong.A3CenvPong(self.render)

# if in test mode:
if not cn.EVAL_MODE and cn.TEST_MODE:
    env = sim.SimulationEnvironment()
    env.runTestMode(cn.TEST_fromTestConsts)
# If in EVAL or TRAIN mode:
else:
    tf.reset_default_graph()

    if not os.path.exists(cn.run_ModelPath):
        os.makedirs(cn.run_ModelPath)


    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    global_rewardEndEpisode = tf.Variable(0,dtype=tf.int32,name='global_rewardEndEpisode',trainable=False)
    trainer = tf.train.RMSPropOptimizer(learning_rate=cn.run_LearningRate, decay=cn.run_decay, epsilon=cn.run_epsilon)
    master_network = AC_Network(cn.run_sSize,cn.run_aSize,'global',None) # Generate global network
    num_workers = cn.run_NumOfWorkers #num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(i,cn.run_sSize,cn.run_aSize,trainer,cn.run_ModelPath,cn.TF_SUMM_PATH,global_episodes,global_rewardEndEpisode))
    saver = tf.train.Saver(max_to_keep=2)

    # If eval mode is on, then just directly use the network to run some
    # environments to completion. Set the right pars in globalConst.py
    if cn.EVAL_MODE:
        save = False
        verbose = True
        model_path =  './LogsOfRuns/' + cn.EVAL_FOLDER + "/model"
        #print(model_path).
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
           # Start 1 EvalWorker
            evalWorker = EvalWorker(cn.ENV_IS_RARM,cn.run_actionSpace,master_network,sess,save,verbose,cn.run_MaxEpisodeLenght)
            evalWorker.playNgames(cn.EVAL_NR_OF_GAMES)
    # This happens in the learning setting, here learning is performed on multiple threads.
    else:
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
        #    workers[0].work(cn.run_MaxEpisodeLenght,cn.run_Gamma,master_network,sess,coord,saver)
        #
            worker_threads = []
        #     Loop the workers and start a thread for each
            for worker in workers:
                # define a function that executes the work method of each worker
                worker_work = lambda: worker.work(cn.run_MaxEpisodeLenght ,cn.run_Gamma,master_network,sess,coord,saver)
                # execute the function in a new thread
                t = threading.Thread(target=(worker_work))
                # start thread
                t.start()
                # keep track of the threads
                worker_threads.append(t)
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


