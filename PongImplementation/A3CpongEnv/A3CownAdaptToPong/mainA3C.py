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

# Processes Doom screen image to produce cropped and resized image.
def process_frame(frame):
    #s = frame[10:-10,30:-30]
    s = rgb2gray(frame)
#    s = skimage.transform.resize(s,[84,84])
    s = scipy.misc.imresize(s,[84,84]) # This was the line in the original github repo but it was depricated
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

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
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.imageIn,num_outputs=16,
                kernel_size=[8,8],stride=[4,4],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv1,num_outputs=32,
                kernel_size=[4,4],stride=[2,2],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)

            #Recurrent network for temporal dependencies

            lstm_cell = tf.nn.rnn_cell.LSTMCell(256,state_is_tuple=True,name='basic_lstm_cell')
            #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256,state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(rnn_out,1,
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
                #TODO deze entropy parameter tunen en bij de andere hyperpars zetten
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.1
                self.adv_sum = tf.reduce_sum(self.advantages)

                #Get gradients from local network using local losses
                # Get local network pars
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                # Calculate gradients of the local loss with respect to the local network pars
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                # This line clips the gradients! preventing a to large update
                #TODO deze clipping parameter tunen, en bij de andere hyperpars zetten
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)

                #Apply local gradients to global network
                # This is the important global network update step!
                # it is executed by the trainer
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))

class Worker():
    def __init__(self,name,s_size,a_size,trainer,model_path,global_episodes,global_rewardEndEpisode, global_episodeFrames):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.global_rewardEndEpisode = global_rewardEndEpisode
        self.global_episodeFrames = global_episodeFrames
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        # write stats of the progress of this worker
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        # Now this network is still epmty
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
        # Copy a list of global network parameters to self.update_local_ops
        self.update_local_ops = update_target_graph('global',self.name)
        # this is correct for Pong
        self.actions = [1,2,3]
        # Specify the environment that the agent will operate in.
        self.env = gym.make('Pong-v0')

    # This function trains the worker on the just completed rollout and at the
    # end it updates the global network.
    # The rollout consists of expierences like s,a,r,s' (but not exactly)
    # This happens periodically after some timesteps. Therefore it needs a bootstrapping value
    # This value is v(se) where se is the end state of the rollout.
    def train(self,global_AC,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"

        # This line adds the bootstrapped value at the end of the array, so it returns: [r1,r2,r3,v(se)]
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        # This line transforms the array of rewards into the right time discounted reward with the bootstrapped value incorporated
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        # This line produces an array of v(si) values.
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        # This step computes the advantage for each timestep but then all at once
        # It computes: adv = r + gamma*v(t+1) - v(t)
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        # This line makes the advantages also dependend on the adventages in the near future in the time discounted sense
        # Maybe this step is from the generalized adventage function algotithm? TODO: find out.
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        # The feeddict is a map from what was just calculted above to the relevant
        # placeholders in the init of this class.
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:rnn_state[0],
            self.local_AC.state_in[1]:rnn_state[1]}
        # Alex: I think this run fuction takes what it should run from the init
        # of this class given the second argument, the feed_dict as input for
        # the placeholders.
        v_l,p_l,e_l,g_n,v_n,adv, apl_g = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.adv_sum,
            self.local_AC.apply_grads], # This line is the UPDATE step
            feed_dict=feed_dict)
        # Take the average of the losses, the gradients and the adventages and return those
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n, v_n, adv/len(rollout)

    def work(self,max_episode_length,gamma,global_AC,sess,coord,saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                # set self.update_local_ops to global NN vars
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                # init the env and states
                s = self.env.reset()
                episode_frames.append(s)
                s = process_frame(s)
                rnn_state = self.local_AC.state_init

                # This while runs until the end of the episode or until
                # max timesteps has been reached. Then done = True
                done = False
                while not done:
                    # Only render one thread and not all
                    if self.number == 0:
                        self.env.render()
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out],
                        feed_dict={self.local_AC.inputs:[s],
                        self.local_AC.state_in[0]:rnn_state[0],
                        self.local_AC.state_in[1]:rnn_state[1]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    # Take one step in env using the chosen action a
                    s1,r,d, i = self.env.step(self.actions[a])
                    episode_frames.append(s1)
                    s1 = process_frame(s1)

                    # Store the experience to the buffer
                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    episode_values.append(v[0,0])

                    # bookkeeping
                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 20 and d != True:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        # v1 is the value estimate usted to bootstrap
                        v1 = sess.run(self.local_AC.value,
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                        # Here the T timesteps of rollout a.k.a the episode_buffer
                        # are used to calculate the loss values and the train function
                        # internally updates the global network pars
                        v_l,p_l,e_l,g_n,v_n, adv = self.train(global_AC,episode_buffer,sess,gamma,v1)
                        # empty the rollout buffer again
                        episode_buffer = []
                        # This stores the network parameters of the global NN
                        # to the variable: self.update_local_ops
                        # and also updates the local_AC to the global pars
                        sess.run(self.update_local_ops)
                    # If the episode terminates or the max steps has been reached
                    # Then the episode (and so this while loop) terminates.
                    if  episode_step_count >= max_episode_length - 1 or d == True:
                        break
                # Now that the episode has terminated, store the relevant
                # statistics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                #Alex added: Store a global end of episode reward to print it
                sess.run(self.global_rewardEndEpisode.assign(int(episode_reward)))
                sess.run(self.global_episodeFrames.assign(int(episode_step_count)))

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    # internally the train function updates the global_AC pars
                    v_l,p_l,e_l,g_n,v_n,adv = self.train(global_AC,episode_buffer,sess,gamma,0.0)


                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    # this saves the sess variable of only the frist worker
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print("Saved Model")

                    # Create a tf.summary
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Advantage', simple_value=float(adv))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                # increment the global episode counter
                sess.run(self.increment)
                # increment internal episode counter
                episode_count += 1
                # loop back to start the new episode


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

#Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')


global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
# Alex added: This is used to print the current reward at the end of an episode
global_rewardEndEpisode = tf.Variable(0,dtype=tf.int32,name='global_rewardEndEpisode',trainable=False)
global_episodeFrames = tf.Variable(0,dtype=tf.int32,name='global_rewardEndEpisode',trainable=False)

trainer = tf.train.RMSPropOptimizer(learning_rate=learningRate, decay=0.99, epsilon=0.1)
master_network = AC_Network(s_size,a_size,'global',None) # Generate global network

workers = []
# Create worker classes
for i in range(num_workers):

workers.append(Worker(i,s_size,a_size,trainer,model_path,global_episodes,global_rewardEndEpisode,global_episodeFrames))
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
        stepCount = sess.run(global_episodeFrames)
        print("Episodes", gs1, 'one for ', (time()-s)/(gs1-gs), "\n" + "Reward at end of episode: ",lastRewardEndOfEp, "# frames this episode: ", stepCount)

        gs = gs1
    coord.join(worker_threads)





