[![arXiv](http://img.shields.io/badge/arXiv-2001.09136-B31B1B.svg)](https://arxiv.org/abs/2107.00606)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 

<h1 align="center">  AdaSCoRe "Adaptive Social Controller with Reinforcement"
</h1>


<p align="center">
  <img src=/readme_images/IROS2024_2634.jpg alt="Alternative text" width="650"/>
</p>

This is the official repo for the IROS 2024 paper "Adaptive Social Force Window Planner with Reinforcement Learning" (https://arxiv.org/abs/2404.13678). The project aims at developing an adaptive social controller algorithm to tackle different social scenarios with people moving. Our system can be declined in different tasks coupling standard navigation algorithms (DWA) and a Reinforcement Learning agent. The RL agent can be trained end-to-end to learn different policies: robot direct commands ('EndRl' task), cost weights ('SocialForceWindow' task), and social costmap parameters ('SocialControllerCostmap'), and new future tasks. 

We use the PIC4rl_gym (https://github.com/PIC4SeR/PIC4rl_gym) as the ROS 2 gym environment to train Deep Reinforcement Learning agents in a Gazebo simulation, designing people configuration with the HuNavSim framework (https://github.com/robotics-upo/hunav_sim).


<p align="center">
  <img src=/readme_images/social_nav_test.png alt="Alternative Text" width="550"/>
</p>



**AdaSCoRe files in pic4rl training package:**
- **pic4rl.launch.py** (instanciate the agent and start the main training loop based on the parameters defined in the config file)
- **jackal_social.launch.py** (launch the Gazebo simulation)
- **bringup.launch.py** (launch the Nav2 stack using the navigation.launch.py file and the localization)

**ROS Nodes:**
- social_controller_adascore 

**Config files:**
- main_param.yaml (simulation, sensors, topics, policy selection, params update frequency)
- training_params.yaml (rl training settings)
- agents_envs (agent and environment settings)
- nav_params (navigation settings)

**COMMANDS:**
- **terminal 1: launch gazebo simulation**
ros2 launch adascore jackal_social.launch.py
- **terminal 3: launch nav2**
ros2 launch adascore bringup.launch.py
- **terminal 2: start trainer**
ros2 launch adascore pic4rl.launch.py 

**Installation**

- use the vcs tool to clone the repo 
```
cd ~/ros2_ws/src
curl
https://raw.githubusercontent.com/maurom3197/ada_score.git/humble/adascore.repos | vcs import src
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
mv <path_to_adascore_pkg>/adjustments/env-hooks .
mv <path_to_adascore_pkg>/CMakeLists.txt .
``` 
- if you want to use the modified version of nav2 dwb controller, you need to build the nav2_dwb_controller pkg removing the COLCON_IGNORE file from the nav2_adascore folder
```
rm <path_to_nav2_adascore_pkg>/COLCON_IGNORE
colcon build --symlink-install
```

**Tested software versions**
- ROS2 Humble
- Nav2 Humble branch
- TensorFlow 2.10.x
- Gazebo 11.0.0
- Ubuntu 22.04
- pic4rl 


# Citations
This repository is intended for scientific research purposes.
If you want to use this code for your research, please cite our work ([Adaptive Social Force Window Planner with Reinforcement Learning](https://arxiv.org/abs/2404.13678)).

```
@article{martini2024adaptive,
  title={Adaptive Social Force Window Planner with Reinforcement Learning},
  author={Martini, Mauro and P{\'e}rez-Higueras, No{\'e} and Ostuni, Andrea and Chiaberge, Marcello and Caballero, Fernando and Merino, Luis},
  journal={arXiv preprint arXiv:2404.13678},
  year={2024}
}
```


# References
```
@article{perez2023hunavsim,
  title={Hunavsim: A ros 2 human navigation simulator for benchmarking human-aware robot navigation},
  author={P{\'e}rez-Higueras, No{\'e} and Otero, Roberto and Caballero, Fernando and Merino, Luis},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}
```
```
@inproceedings{martini2023pic4rl,
  title={Pic4rl-gym: a ros2 modular framework for robots autonomous navigation with deep reinforcement learning},
  author={Martini, Mauro and Eirale, Andrea and Cerrato, Simone and Chiaberge, Marcello},
  booktitle={2023 3rd international conference on computer, control and robotics (ICCCR)},
  pages={198--202},
  year={2023},
  organization={IEEE}
}
```

# Acknowledgements
[This work has been realized thanks to a joint effort by researchers at PIC4SeR Centre for Service Robotics at Politecnico di Torino (https://pic4ser.polito.it/) and the Service Robotics Lab of the Pablo de Olavide University, Sevilla (https://robotics.upo.es/).]
