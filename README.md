# APPLR_social_nav

## Description of the project
This project aims at developing an adptative social navigation algorithm. Inspired by the parameter learning approach presented in APPLR (https://arxiv.org/abs/2011.00397), our system social navigation behaviour expressed in the reward signal of the DRL agent. We use the PIC4rl_gym as the ROS2 gym environment to train Deep Reinforcement Learning agents in a Gazebo simulation.

**APPLR files in pic4rl training package:**
- **pic4rl.launch.py** (instanciate the agent and start the main training loop based on the parameters defined in the config file)
- **jackal_social.launch.py** (launch the Gazebo simulation)
- **bringup.launch.py** (launch the Nav2 stack using the navigation.launch.py file and the localization)

**ROS Nodes:**
- social_controller_applr 

**Config files:**
- main_param.yaml (simulation, sensors, topics, policy selection, params update frequency)
- training_params.yaml (rl training settings)
- agents_envs (agent and environment settings)
- nav_params (navigation settings)

**COMMANDS:**
- **terminal 1: launch gazebo simulation**
ros2 launch applr_social_nav jackal_social.launch.py
- **terminal 3: launch nav2**
ros2 launch applr_social_nav bringup.launch.py
- **terminal 2: start trainer**
ros2 launch applr_social_nav pic4rl.launch.py 

**Install**

- use the vcs tool to clone the repo 
```
cd ~/ros2_ws/src
curl
https://raw.githubusercontent.com/maurom3197/APPLR_social_nav.git/humble/applr_social_nav.repos | vcs import src
```
- install lightsfm ([view the repo](https://github.com/robotics-upo/lightsfm) for more details)
- install packages required for tf2rl (tf2rl/setup.py)
- install dependencies and build
```
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
```
- modify the hunav_gazebo_wrapper pkg to use the env-hooks to load the gazebo models
```
cd ~/ros2_ws/src/hunav_gazebo_wrapper
mv <path_to_applr_pkg>/adjustments/env-hooks .
mv <path_to_applr_pkg>/CMakeLists.txt .
``` 




**Tested software versions**
- ROS2 Humble
- Nav2 Humble branch
- TensorFlow 2.10.x
- Gazebo 11.0.0
- Ubuntu 22.04
- pic4rl 
