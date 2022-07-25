#!/usr/bin/env python3

import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.initializers import RandomUniform, glorot_normal, HeUniform, GlorotUniform
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import backend as K
import numpy as np

class ActorVW(tf.keras.Model):
    def __init__(self, state_shape, action_dim, max_action=(0.4,1.), min_action=(0.,-1.), units=(256, 256), name="Actor"):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        #self.l3 = Dense(units[2], name="L3")
        #Output Layer
        self.v_out = Dense(1, activation = 'sigmoid')
        self.w_out = Dense(1, activation = 'tanh')

        self.max_action = tf.cast(max_action, dtype = tf.float32)
        #self.min_action = tf.cast(min_action, dtype = tf.float32)

        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32)))

    def call(self, inputs):
        features = tf.nn.relu(self.l1(inputs))
        features = tf.nn.relu(self.l2(features))
        #features = tf.nn.relu(self.l3(features))
        
        Linear_velocity = self.v_out(features)*self.max_action[0]
        Angular_velocity = self.w_out(features)*self.max_action[1]
        action = concatenate([Linear_velocity, Angular_velocity])
        return action

class Actor(tf.keras.Model):
    def __init__(self, state_shape, action_dim, max_action=(0.4,1.), min_action=(0.,-1.),units=(256, 256), name="Actor"):
        super().__init__(name=name)

        # Base Layers
        self.base_layers = []
        for i in range(len(units)):
            unit = units[i]
            self.base_layers.append(Dense(unit, activation='relu'))

        # Output Layer
        self.out_layer = Dense(action_dim, activation = 'tanh')

        self.max_action = tf.cast(max_action, dtype = tf.float32)
        self.min_action = tf.cast(min_action, dtype = tf.float32)
        
        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32)))

    def call(self, features):
        for cur_layer in self.base_layers:
            features = cur_layer(features)
        action = self.out_layer(features)*self.max_action
        return action


class Critic(tf.keras.Model):
    def __init__(self, state_shape, action_dim, units=(256, 256), name="Critic"):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        #self.l3 = Dense(units[2], name="L3")
        self.lout = Dense(1, name="Lout")

        dummy_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=[1, action_dim], dtype=np.float32))
        with tf.device("/cpu:0"):
            self(dummy_state, dummy_action)

    def call(self, states, actions):
        features = tf.concat((states, actions), axis=1)
        features = tf.nn.relu(self.l1(features))
        features = tf.nn.relu(self.l2(features))
        #features = tf.nn.relu(self.l3(features))
        values = self.lout(features)
        return tf.squeeze(values, axis=1)

class CriticQ(tf.keras.Model):
    def __init__(self, state_shape, action_dim, critic_units=(256, 256), name='Critic'):
        super().__init__(name=name)
	
        self.model_name = name
        self.state_input = Input(shape=state_shape)
        self.action_input = Input(shape=(action_dim,))
        self.base_layers = []
        for i in range(len(critic_units)):
            unit = critic_units[i]
            self.base_layers.append(Dense(unit, activation='relu'))
        self.out_layer = Dense(1, name="Q", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        self(dummy_state, dummy_action)

    def call(self, states, actions):
        features = tf.concat((states, actions), axis=1)
        for cur_layer in self.base_layers:
            features = cur_layer(features)
        values = self.out_layer(features)
        return tf.squeeze(values, axis=1)

    def model(self):
        return tf.keras.Model(inputs = [self.state_input, self.action_input], outputs = self.call(self.state_input, self.action_input), name = self.model_name)

class ConvCriticQ(tf.keras.Model):
    def __init__(self, state_shape, action_dim, num_layers=2, critic_units=(256, 256), 
                 num_conv_layers=4, conv_filters=(32,64), filt_size = (3,3), name='qf'):
        super().__init__(name=name)

        self.model_name = name
        self.state_input = Input(shape=state_shape)
        self.action_input = Input(shape=(action_dim,))

        self.k_initializer = HeUniform()
        self.conv_layers = []
        for layer_idx in range(2):
            # TODO: Check padding and activation
            stride = 2 if layer_idx == 0 else 1 # check: correct
            self.conv_layers.append(
                Conv2D(conv_filters[0], kernel_size=filt_size, strides=(stride, stride), padding='valid',
                       kernel_initializer=self.k_initializer, activation='relu'))
        
        self.conv_layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        
        for layer_idx in range(2):
            # TODO: Check padding and activation
            stride = 2 if layer_idx == 0 else 1 # check: correct
            self.conv_layers.append(
                Conv2D(conv_filters[1], kernel_size=filt_size, strides=(stride, stride), padding='valid',
                       kernel_initializer=self.k_initializer, activation='relu'))
        
        self.conv_layers.append(GlobalAveragePooling2D())

        self.base_layers = []
        for i in range(num_layers):
            unit = critic_units[i]
            self.base_layers.append(Dense(unit, activation='relu'))
        self.out_layer = Dense(1, name="Q", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        self(dummy_state, dummy_action)

    def call(self, states, actions):
        features = states
        for conv_layer in self.conv_layers:
            features = conv_layer(features)
        features = tf.concat((features, actions), axis=1)
        for cur_layer in self.base_layers:
            features = cur_layer(features)
        values = self.out_layer(features)
        return tf.squeeze(values, axis=1)

    def model(self):
        return tf.keras.Model(inputs = [self.state_input, self.action_input], outputs = self.call(self.state_input, self.action_input), name = self.model_name)

class ConvMixCriticQ(tf.keras.Model):
    def __init__(self, state_shape, action_dim, num_layers=2, critic_units=(256, 256), 
                 num_conv_layers=4, conv_filters=(32,64), filt_size = (3,3), name='qf'):
        super().__init__(name=name)

        self.model_name = name
        self.state_input = Input(shape=state_shape)
        self.action_input = Input(shape=(action_dim,))

        self.image_shape = (112,112,1,)
        self.state_info_shape=2

        self.k_initializer = HeUniform()
        self.conv_layers = []
        for layer_idx in range(2):
            # TODO: Check padding and activation
            stride = 2 if layer_idx == 0 else 1 # check: correct
            self.conv_layers.append(
                Conv2D(conv_filters[0], kernel_size=filt_size, strides=(stride, stride), padding='valid',
                       kernel_initializer=self.k_initializer, activation='relu'))
        
        self.conv_layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        
        for layer_idx in range(2):
            # TODO: Check padding and activation
            stride = 2 if layer_idx == 0 else 1 # check: correct
            self.conv_layers.append(
                Conv2D(conv_filters[0], kernel_size=filt_size, strides=(stride, stride), padding='valid',
                       kernel_initializer=self.k_initializer, activation='relu'))
        
        self.conv_layers.append(GlobalAveragePooling2D())

        self.base_layers = []
        for i in range(num_layers):
            unit = critic_units[i]
            self.base_layers.append(Dense(unit, activation='relu'))
        self.out_layer = Dense(1, name="Q", activation='linear')

        dummy_state = tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, action_dim], dtype=np.float32))
        self(dummy_state, dummy_action)

    def call(self, states, actions):
        b = tf.shape(states)[0]
        state_info = states[:,:self.state_info_shape]
        img_array = states[:,self.state_info_shape:]
        features = tf.reshape(img_array, (b,self.image_shape[0],self.image_shape[1],self.image_shape[2]))

        for conv_layer in self.conv_layers:
            features = conv_layer(features)
        features = tf.concat((state_info,features,actions), axis=1)
        #features = tf.concat((features, actions), axis=1)
        for cur_layer in self.base_layers:
            features = cur_layer(features)
        values = self.out_layer(features)
        return tf.squeeze(values, axis=1)

    def model(self):
        return tf.keras.Model(inputs = [self.state_input, self.action_input], outputs = self.call(self.state_input, self.action_input), name = self.model_name)