
My phd AI project on (Xiaor Geek car) robotics  
based on Multimodal Transformers (Natural-language, Computer vision, Lidar point cloud)
* Environments: real-world robotics, GAZEBO simulation


Repo:
1. ~/catkin_ws/src/mobile_manipulator_body - ROS package of virtual GAZEBO simulation of robot (diff_drive with odom, camera, lidar(SLAM,amcl))
2. ~/catkin_ws/src/xrrobot_project/xrrobot - ROS package of xrrobot (Xiaor Geek SLAM car)  
3. ~/pythonprogv2/phd_xiaor_project - Python scripts source (this repo) 

v07.03.2024
Ubuntu 20.04.6 LTS
ROS noetic
Python 3.8.10
Torch cuda 11.8


Build catkin_ws project:
cd ~/catkin_ws
source ./devel/setup.bash
catkin_make



Start-up: 
1. Gazebo Virtual environment experiment (camera experiment):
    (bash-based)
    For every node:
    cd ~/catkin_ws
    source ./devel/setup.bash
    Independent nodes:
    1. roslaunch mobile_manipulator_body base_gazebo_control.launch
    2. rviz -d ~/catkin_ws/src/mobile_manipulator_body/camera_view.rviz
    3. rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/robot_base_velocity_controller/cmd_vel
    4. python3 reset_env.py
    5. python3 reward_publisher.py
    6. python3 decision_transformer2.py
    7. start tensorboard by clicking the button in import side

2. Real-world robot experiment:
    0. export ROS_MASTER_URI=http://172.16.1.150:11311 -> ~/.bashrc
    1. Turn on xrrobot (connect to network)
    2. roscore
    xrrobot connection: ssh xrrobot@172.16.1.152
    3. xrrobot: roslaunch xrrobot bringup.launch
    4.rosrun teleop_twist_keyboard teleop_twist_keyboard.py
    5. xrrobot: roslaunch xrrobot camera.launch
    6. rosrun rqt_image_view rqt_image_view
    7. launch navigation (move_base) with obstacle map supply (amcl) or mapping (SLAM)
    (SLAM + move_base): 
    * roslaunch xrrobot lidar_slam4.launch (SLAM + move_base)
    * rosrun map_server map_saver -f ./maps/2A724_april
    (amcl + move_base):
    * roslaunch xrrobot navigate4.launch
    * rosrun map_server map_server ./maps/2A724new.yaml

    8. rviz -d ./navigate.rviz

    rosservice call /start_motor "{}" - Lidar service (sometimes helps when Lidar not publishing)

3. GAZEBO virtual env (camera-lidar experiment): 
    1. Launch environment
    * roslaunch mobile_manipulator_body 2A724_empty.launch
    * roslaunch mobile_manipulator_body 2A724_m.launch
    * roslaunch mobile_manipulator_body base_gazebo_control.launch
    2. launch navigation (move_base) with obstacle map supply (amcl) or mapping (SLAM)
    (SLAM + move_base): 
    * roslaunch mobile_manipulator_body lidar_slam4.launch (SLAM + move_base)
    (amcl + move_base):
    * roslaunch mobile_manipulator_body navigate4.launch
    ** roscd mobile_manipulator_body
    ** rosrun map_server map_server ./maps/2A724x3.yaml
    3. (optional) keyboard control:
    * rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/rob/cmd_vel
    4. (optional) Map visualization:
    * rviz -d ./amcl_navigate.rviz
    * rviz -d ./navigate.rviz
    5. (optional) camera view:
    * rosrun rqt_image_view rqt_image_view
    * rviz -d ./camera_view.rviz






watch -n 1 nvidia-smi
tensorboard --logdir=/home/renas/pythonprogv2/phd_xiaor_project --port=6007



weights files:
renas3_last.pt - renas_train_multiproc3 weights file LR 10e-5 90 epochs loss 0.1519
renas3_ag.pt - renas_train_multiproc3 weights file LR 10e-5 120 epochs loss 0.1488


renas3.1_env.pt - renas_train_multiproc3.1 weights file LR 10e-5 env 70 epoch loss about 45.0
renas3.1_pink_gates.pt renas_train_multiproc3.1 weights file LR 10e-5 40 epoch loss 48.06


