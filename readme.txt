Gazebo based robotic (Xiaor Geek) simulation project for Transformer experiments

1. ~/catkin_ws/src/mobile_manipulator_body - ROS package
2. ~/pythonprogv2/phd_xiaor_project - Python scripts source

1.1. ./launch/base_gazebo_control.launch
A file to start ga project's gazebo simulation 
1.2. ./meshes 
Visual robot represenations
1.3. ./sdf
sdf format additions to gazebo
1.4. ./urdf/robot_base.urdf
Robot description, logic models and plugins



v30.01.2024
Ubuntu 20.04.6 LTS
ROS noetic
Python 3.8.10
Torch cuda 11.8


Build catkin_ws project:
cd ~/catkin_ws
source ./devel/setup.bash
catkin_make



Start-up: 
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
7. start tensorboard by clicking the button in decision_transformer2.py import side

watch -n 1 nvidia-smi
tensorboard --logdir=/home/renas/pythonprogv2/phd_xiaor_project --port=6007



weights files:
renas3_last.pt - renas_train_multiproc3 weights file LR 10e-5 90 epochs loss 0.1519
renas3_ag.pt - renas_train_multiproc3 weights file LR 10e-5 120 epochs loss 0.1488
renas3_real.pt - renas_train_multiproc3 weights for real_pink_gates dataset LR 1oe-5 90 epochs loss 0.1472
renas3.1_env.pt - renas_train_multiproc3.1 weights file LR 10e-5 env 70 epoch loss about 45.0
renas3.1_pink_gates.pt renas_train_multiproc3.1 weights file LR 10e-5 40 epoch loss 48.06


