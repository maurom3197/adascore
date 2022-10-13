# APPLR_social_nav

## Description of the project
This project aims at developing an adptative social navigation algorithm. Inspired by the parameter learning approach presented in APPLR (https://arxiv.org/abs/2011.00397), our system social navigation behaviour expressed in the reward signal of the DRL agent. We use the PIC4rl_gym as the ROS2 gym environment to train Deep Reinforcement Learning agents in a Gazebo simulation.

APPLR files in pic4rl training package:
- pic4rl_trainer.py (instanciate the agent and start the main training loop)
- pic4rl_training_nav2.py (select and define the agent, define action and state spaces)
- pic4rl_environment_nav2.py (interact with the agent with the 'step()' and 'reset()' functions, compute observation and rewards, publish goal for the navigator, call params services of the controller node as action, check end of episode condition and the navigation status)
- generic_sensor.py: start the necessary sensors topic subscription
- sensors: contain all the necessary method to preprocess sensor data
- navigator.py: an interface ROS2 Node to easy handle and monitor the navigation during the simulation.

ROS Nodes:
- pic4rl_training(pic4rl_environment(Node))
- Navigator(Node)

Config files: 
- main_param.yaml (simulation, sensors, topics, policy selection, params update frequency)
- training_params.yaml (rl training settings)

COMMANDS:
# terminal 1: launch gazebo simulation
ros2 launch gazebo_sim simulation.launch.py

# terminal 2: start trainer 
ros2 run pic4rl pic4rl_trainer

# terminal 3: launch nav2
 ros2 launch pic4nav nav.launch.py 

 - TO DO -
In the .bashrc export the gazebo models path:
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:'...~/src/APPLR_social_nav/gazebo_sim'
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:'...~/src/APPLR_social_nav/gazebo_sim/models'

 - Tested software versions -
ROS2 Foxy
Nav2 Foxy branch
TensorFlow 2.6.x
Keras 2.6.x

Try to build tf2rl setup.py:
- go in the directory: ~/APPLR_social_nav/training/tf2rl
- pip install .

or install manually the packages in setup.py at ~/APPLR_social_nav/training/tf2rl/setup.py

