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
from tensorflow.keras.applications import MobileNetV3Small
import numpy as np

class Backbone(Model):
	def __init__(self, height = 224, width  = 224,   name = 'backbone', **kwargs):
		super(Backbone, self).__init__(**kwargs)

		self.model_name = name

		# image shape
		self.height = height
		self.width = width
		self.n_frames = 1
		self.image_shape = (1, self.height, self.width, 3,)

		#Layers definition
		#Input Layer
		self.image_input = Input(shape=(self.height, self.width, 3,))  

		#Backbone
		self.backbone = MobileNetV3Small(
			input_shape=(224,224,3), alpha=1.0, minimalistic=False, include_top=False,
			weights='imagenet', input_tensor=self.image_input, pooling=None,
			include_preprocessing=True)

		dummy_image = tf.constant(
		 	np.zeros(shape= self.image_shape, dtype=np.float32))

		self.model()
		self(dummy_image)

	def call(self, image):
		#x = tf.keras.applications.mobilenet.preprocess_input(image)
		x = self.backbone(image)
		xf = GlobalAveragePooling2D()(x)
		return xf

	def model(self):
		image_input = Input(shape=(self.height, self.width, 3,))
		return Model(inputs = [image_input], outputs = self.call(image_input), name = self.model_name)

class ActorCNNetwork(Model):
	def __init__(self, max_linear_velocity, max_angular_velocity, features_shape = 1024, goal_shape = 1, lr = 0.00025, fc1_dims = 400, fc2_dims = 300, fc3_dims = 300,  name = 'actor', **kwargs):
		super(ActorCNNetwork, self).__init__(**kwargs)

		self.model_dir_path = os.path.dirname(os.path.realpath(__file__))
		self.model_dir_path = self.model_dir_path.replace(
		    '/pic4rl/pic4rl/pic4rl',
		    '/pic4rl/pic4rl/models/agent_camera_model')

		#Learning rate and optimizer
		self.lr = lr
		self.optimizer = Adam(learning_rate = self.lr)
		self.model_name = name

		#Velocity limits
		self.max_linear_velocity = max_linear_velocity
		self.max_angular_velocity = max_angular_velocity

		self.goal_shape = (goal_shape,)
		self.features_shape = (features_shape,)

		#Layers definition
		#Input Layer
		self.goal_input = Input(shape=(goal_shape,))
		self.features_input = Input(shape=(features_shape,))


		#Hidden Layer
		self.k_initializer = HeUniform()
		self.fc1 = Dense(fc1_dims, activation='relu', kernel_initializer = self.k_initializer)
		self.fc2 = Dense(fc2_dims, activation='relu', kernel_initializer = self.k_initializer)
		self.fc3 = Dense(fc3_dims, activation='relu', kernel_initializer = self.k_initializer)

		#Output Layer
		self.output_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
		self.linear_out = Dense(1, activation = 'sigmoid', kernel_initializer = self.output_initializer)
		self.angular_out = Dense(1, activation = 'tanh', kernel_initializer = self.output_initializer)
		
		dummy_goal = tf.constant(
		 	np.zeros(shape =(1,) + self.goal_shape, dtype=np.float32))
		dummy_features = tf.constant(
		 	np.zeros(shape = (1,) + self.features_shape, dtype=np.float32))

		self.model()
		self(dummy_goal, dummy_features)

	def call(self, goal, features):

		xf = self.fc1(features)
		xf = self.fc2(xf)
		x = concatenate([xf, goal])

		out = self.fc3(x)

		Linear_velocity = self.linear_out(out)*self.max_linear_velocity
		Angular_velocity = self.angular_out(out)*self.max_angular_velocity
		action = concatenate([Linear_velocity, Angular_velocity])

		return action

	def model(self):

		return Model(inputs = [self.goal_input, self.features_input], outputs = self.call(self.goal_input, self.features_input), name = self.model_name)

class CriticCNNetwork(Model):
	def __init__(self, goal_shape = 1, features_shape = 1024, lr = 0.001, fc1_dims = 400, fc2_dims = 300, fc3_dims = 300,  name = 'critic', **kwargs):
		super(CriticCNNetwork, self).__init__(**kwargs)

		self.model_dir_path = os.path.dirname(os.path.realpath(__file__))
		self.model_dir_path = self.model_dir_path.replace(
		    '/pic4rl/pic4rl/pic4rl',
		    '/pic4rl/pic4rl/models/agent_camera_model')

		#Learning rate, optimizer, loss, name
		self.lr = lr
		self.optimizer = Adam(learning_rate = self.lr)
		self.loss = tf.keras.losses.MeanSquaredError()
		self.model_name = name
		#input shape		
		self.features_shape = (features_shape,)
		self.goal_shape = (goal_shape,)

		# Input Layers
		self.goal_input = Input(shape=(goal_shape,))
		self.features_input = Input(shape=(features_shape,))
		self.action_input = Input(shape=(2,))


		#Hidden Layer
		self.k_initializer = HeUniform()
		self.fc1 = Dense(fc1_dims, activation='relu',  kernel_initializer = self.k_initializer)
		self.fc2 = Dense(fc2_dims, activation='relu',  kernel_initializer = self.k_initializer)
		self.fc3 = Dense(fc3_dims , activation='relu', kernel_initializer = self.k_initializer)

		#Output Layer
		self.output_initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
		self.out = Dense(1, activation='linear', kernel_initializer = self.output_initializer)
		
		dummy_goal = tf.constant(
		 	np.zeros(shape = (1,) + self.goal_shape, dtype=np.float32))
		dummy_features = tf.constant(
		 	np.zeros(shape = (1,) + self.features_shape, dtype=np.float32))
		dummy_action = tf.constant(
			np.zeros(shape = (1, 2,), dtype=np.float32))

		self(dummy_goal, dummy_features, dummy_action)
		self.model()

	def call(self, goal, features, action):
		xf = self.fc1(features)
		xf = self.fc2(xf)
		x = concatenate([xf, goal])
		x_conc = concatenate([x, action])
		x2 = self.fc3(x_conc)

		q = self.out(x2)

		return tf.squeeze(q, axis=1)

	def model(self):

		return Model(inputs = [self.goal_input, self.features_input, self.action_input], outputs = self.call(self.goal_input, self.features_input, self.action_input), name = self.model_name)  
