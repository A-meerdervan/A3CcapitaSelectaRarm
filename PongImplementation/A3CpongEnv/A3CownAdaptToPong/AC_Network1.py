# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 15:20:32 2018

@author: Alex
"""


import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
#---- End imports ----

class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape = [None, s_size],dtype=tf.float32, name='inputState')
            hidden1 = slim.fully_connected(self.inputs,256,activation_fn=tf.nn.relu)
            # TODO: maybe slim.flatten() gebruiken voor je de layer als input geeft
            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(hidden1,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=self.normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(hidden1,1,
                activation_fn=None,
                weights_initializer=self.normalized_columns_initializer(1.0),
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
        
        #Used to initialize weights for policy and value output layers
    def normalized_columns_initializer(self,std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer