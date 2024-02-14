FROM nvcr.io/nvidia/tensorrt:23.12-py3 as jammy_tensorflow_210

# Install Tensorflow 2.10
ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/cuda-12.3/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH

RUN ln -s /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudart.so.12 /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudart.so.11.0
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer.so.8 /usr/lib/x86_64-linux-gnu/libnvinfer.so.7
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8 /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7
RUN ln -s /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcublas.so.12 /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcublas.so.11
RUN ln -s /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcublasLt.so.11
RUN ln -s /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcufft.so.11 /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcufft.so.10
RUN ln -s /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcusparse.so.12 /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcusparse.so.11

RUN python3 -m pip install tensorflow==2.10.* tensorflow-probability==0.18.* tensorflow-addons==0.18.* gym[atari]==0.21


FROM jammy_tensorflow_210 as ros2_desktop_full

ARG ROS_DISTRO=humble
ARG DEBIAN_FRONTEND=noninteractive

# Set the locale
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
  && export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get install -y tzdata \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && rm -rf /var/lib/apt/lists/*

# Install common programs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    sudo \
    software-properties-common \
    wget \
    bash-completion \
    build-essential \
    cmake \
    gdb \
    git \
    openssh-client \
    python3-argcomplete \
    python3-pip \
    vim \
    && rm -rf /var/lib/apt/lists/* 

# Install ROS2
RUN sudo add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y --no-install-recommends \
      ros-$ROS_DISTRO-desktop \
      ros-dev-tools \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=$ROS_DISTRO
ENV AMENT_PREFIX_PATH=/opt/ros/$ROS_DISTRO
ENV COLCON_PREFIX_PATH=/opt/ros/$ROS_DISTRO
ENV LD_LIBRARY_PATH=/opt/ros/$ROS_DISTRO/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/ros/$ROS_DISTRO/bin:$PATH
ENV PYTHONPATH=/opt/ros/$ROS_DISTRO/lib/python3.10/site-packages
ENV ROS_PYTHON_VERSION=3
ENV ROS_VERSION=2

# Install Gazebo
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O \
  /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
  http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
  && apt-get update && apt-get install -q -y --no-install-recommends \
    ros-$ROS_DISTRO-gazebo* \
  && rm -rf /var/lib/apt/lists/*

FROM ros2_desktop_full as pic4rl_gym

# Install additional ROS 2 packages

# RUN sed --in-place --expression \
#       '$isource "$ROBOT_WS/install/setup.bash"' \
#       /ros_entrypoint.sh

# # Remove the source, log and build files to clean image
# RUN rm -rf $ROBOT_WS/src \
#     && rm -rf $ROBOT_WS/build \
#     && rm -rf $ROBOT_WS/log

# # Install additional ROS 2 packages
# RUN apt update && apt install -y \
#     ros-$ROS_DISTRO-realsense2-description \
#     ros-$ROS_DISTRO-interactive-marker-twist-server \
#     ros-$ROS_DISTRO-twist-mux \
#     # INSERT HERE
#     && rm -rf /var/lib/apt/lists/*

# # Add user with same UID and GID as your host system
# # (copied from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
# # Switch from root to user
USER $USERNAME

# # Add user to groups to allow access to hardware devices
RUN sudo usermod --append --groups video $USERNAME
RUN sudo usermod --append --groups dialout $USERNAME
RUN sudo usermod --append --groups tty $USERNAME

# # Update all packages
RUN sudo apt update && sudo apt upgrade -y

# # Rosdep update
RUN sudo rosdep init
RUN rosdep update

# # Source the ROS setup file
# RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
# RUN echo "source ${ROBOT_WS}/install/setup.bash" >> ~/.bashrc

# WORKDIR $ROBOT_WS

ARG APPLR_WS=/home/$USERNAME/applr_ws
RUN mkdir -p $APPLR_WS
WORKDIR $APPLR_WS

COPY ./applr_social_nav.repos ./

RUN mkdir src && vcs import src < ./applr_social_nav.repos

# # General ROS development tools
RUN sudo apt update && sudo apt install -y \
    libboost-system-dev \
    python3-colcon-mixin \
  && sudo rm -rf /var/lib/apt/lists/*

# # Install dependencies for building ROBOT_WS
RUN . /opt/ros/$ROS_DISTRO/setup.bash && \
    sudo apt-get update && rosdep install -y \
      --from-paths src \
      --ignore-src \
    && sudo rm -rf /var/lib/apt/lists/*

# # Install additional dependencies that are not managed by rosdep
RUN sudo apt update && sudo apt install -y \
    ros-$ROS_DISTRO-gazebo-ros-pkgs \
    && sudo rm -rf /var/lib/apt/lists/*

# Install tf2rl
RUN python3 -m pip install src/pic4rl_gym/tf2rl

# Install lightsfm
RUN cd src/hunav/lightsfm/ && make && sudo make install

# Add mixins
RUN colcon mixin add default \
  https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml \
  && colcon mixin update default

# # Build APPLR_WS
ARG OVERLAY_MIXINS="release"
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build \
      --mixin $OVERLAY_MIXINS \
      --symlink-install \
      --executor sequential

ENV APPLR_WS $APPLR_WS

# Build applr_social_nav
RUN mkdir -p src/APPLR_social_nav
COPY . src/APPLR_social_nav/

RUN . /opt/ros/$ROS_DISTRO/setup.bash && \
    sudo apt-get update && rosdep install -y \
      --from-paths src \
      --ignore-src \
    && sudo rm -rf /var/lib/apt/lists/*

RUN . ${APPLR_WS}/install/setup.bash && \
    colcon build \
      --mixin $OVERLAY_MIXINS \
      --symlink-install 



