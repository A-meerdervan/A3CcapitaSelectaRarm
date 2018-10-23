# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:48:28 2018

@author: Alex
"""

import numpy as np
import tensorflow as tf

# Alex: added this
from Env.GlobalConstantsA3C import gc # has high level constants
import Env.A3CenvPong # import custom Pong env (no images but 6 numbers as state)
from AC_Network import AC_Network # The network used as brain

import scipy.signal
from skimage.color import rgb2gray # gave an error, had to pip it, then another error then had to pip scikit-image
import skimage

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
        # TODO make global somehow
        self.tfSummaryEpInterval = 5 # was 5
        self.tfSaveModelInterval = 250 # was 250
        self.rolloutLength = 20 # was at 20 for the Dm-mc implementation
        # write stats of the progress of this worker
        # This creates a tensorboard summary writer that writes to tfSummaryPath+ "01" for example
        self.summary_writer = tf.summary.FileWriter(tfSummary_path[2:] +str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        # Now this network is still epmty
        self.local_AC = AC_Network(s_size,a_size,self.name,trainer)
        # Copy a list of global network parameters to self.update_local_ops
        self.update_local_ops = self.update_target_graph('global',self.name)        
        # this is correct for Pong
        self.actions = [1,2,3]
        # Specify the environment that the agent will operate in.
        #self.env = gym.make('Pong-v0')
        self.env = Env.A3CenvPong.A3CenvPong()
        
    # This function trains the worker on the just completed rollout and at the
    # end it updates the global network.
    # The rollout consists of expierences like s,a,r,s' (but not exactly)
    # This happens periodically after some timesteps. Therefore it needs a bootstrapping value
    # This value is v(se) where se is the end state of the rollout.
    def train(self,rollout,sess,gamma,bootstrap_value):
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
        discounted_rewards = self.discount(self.rewards_plus,gamma)[:-1]
        # This line produces an array of v(si) values. 
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        # This step computes the advantage for each timestep but then all at once
        # It computes: adv = r + gamma*v(t+1) - v(t)
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        # This line makes the advantages also dependend on the adventages in the near future in the time discounted sense
        # Maybe this step is from the generalized adventage function algotithm? TODO: find out.
        advantages = self.discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # The feeddict is a map from what was just calculated above to the relevant
        # placeholders in the init of this class.
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages}
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
                #episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                
                # init the env and states
                s = self.env.reset()
                #episode_frames.append(s)
                #s = self.process_frame(s) # not needed when not having frames
                #rnn_state = self.local_AC.state_init
                
                # This while runs until the end of the episode or until 
                # max timesteps has been reached. Then done = True
                done = False
                while not done:
                    # Only render one thread and not all
                    if self.number == 0 and gc.RENDER_SCREEN:
                        self.env.render()
                    #Take an action using probabilities from policy network output.
                    a_dist,v = sess.run([self.local_AC.policy,self.local_AC.value],
                        feed_dict={self.local_AC.inputs:[s]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    # Take one step in env using the chosen action a
                    s1,r,d, i = self.env.step(self.actions[a])
                    # No frames to proces so commented out:
                    #episode_frames.append(s1)
                    #s1 = self.process_frame(s1)
                        
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
                    if len(episode_buffer) == self.rolloutLength and d != True:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        # v1 is the value estimate usted to bootstrap
                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[s]})[0,0]
                        # Here the T timesteps of rollout a.k.a the episode_buffer
                        # are used to calculate the loss values and the train function
                        # internally updates the global network pars
                        v_l,p_l,e_l,g_n,v_n, adv = self.train(episode_buffer,sess,gamma,v1)
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
                
                
                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    # internally the train function updates the global_AC pars
                    v_l,p_l,e_l,g_n,v_n,adv = self.train(episode_buffer,sess,gamma,0.0)
                                
                    
                # Periodically save model parameters, and summary statistics.
                if episode_count % self.tfSummaryEpInterval == 0 and episode_count != 0: #TODO change to 5
                    # this saves the sess variable and so the entire network. 
                    # It is only triggerd by the first worker but holds info on everything
                    if episode_count % self.tfSaveModelInterval == 0 and self.name == 'worker_0': #TODO change to %250
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print("Saved Model at local ep count: ", episode_count)

                    # Create a tf.summary
                    mean_reward = np.mean(self.episode_rewards[-self.tfSummaryEpInterval :])
                    mean_length = np.mean(self.episode_lengths[-self.tfSummaryEpInterval :])
                    mean_value = np.mean(self.episode_mean_values[-self.tfSummaryEpInterval:])
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
                #sess.run(self.increment)
                if self.name == 'worker_0':
                    sess.run(self.increment)
                # increment internal episode counter
                episode_count += 1
                # loop back to start the new episode
        # Pong specific, just to quit the pygame windows so they dont crash
        self.env.quitScreen()
        # Copies one set of variables to another.
    # Used to set worker network parameters to those of global network.
    def update_target_graph(self,from_scope,to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    
        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder
    
    # Processes Doom screen image to produce cropped and resized image. 
    def process_frame(self,frame):
        #s = frame[10:-10,30:-30]
        s = rgb2gray(frame)
        s = skimage.transform.resize(s,[84,84])
        #s = scipy.misc.imresize(s,[84,84]) # This was the line in the original github repo but it was depricated
        s = np.reshape(s,[np.prod(s.shape)]) / 255.0
        return s
    
    # Discounting function used to calculate discounted returns.
    def discount(self,x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]