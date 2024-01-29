import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import numpy as np
import time
from collections import deque
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
import os
from std_msgs.msg import String
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
import random
import math
import threading
from torchvision import transforms
import cv2

'''
Node to gather (state-action) trajectories with fixed number in a trajectory,
restart the env itself, change cube's and sphere's spawn place
Work without reset_env.py, reward_publisher.py
States and actions buffer are separated, single image on a state, 
3 channels im only,
std, mean preprocess for pretrained models
(Used to be imported)
'''

class TrajectoryBuffer:
    def __init__(self, image_topic = '/image_raw', cmd_vel_topic = 'robot_base_velocity_controller/cmd_vel',buffer_size=10, im_resolution = (640,480), im_preproc = True, num_transitions=100, always=True, reset_environment=True):
        self.always = always
        self.num_transitions = num_transitions
        self.im_resolution = im_resolution
        self.buffer_size = buffer_size
        self.states_buffer = deque([[]], maxlen=buffer_size)
        self.actions_buffer = deque([[]], maxlen=buffer_size)
        self.action = deque([(0, 0)], maxlen=1)
        self.gather = True
        self.im_preproc = im_preproc
        if im_preproc:
            self.preprocess = transforms.Normalize(    
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        self.reset_environment = reset_environment
        #self.preprocess = transforms.Compose([
        #    transforms.Resize((224, 224)),  # Resize to match MobileNetV2 input size
        #    transforms.ToTensor(),           # Convert to tensor
        #    transforms.Normalize(            # Normalize with ImageNet stats
        #        mean=[0.485, 0.456, 0.406],
        #        std=[0.229, 0.224, 0.225]
        #        ),
        #    ])

        # Initialize the ROS node and subscribe to the image topic
        rospy.init_node('traj_gather_node', anonymous=True)
        rospy.Subscriber(image_topic, Image, self.callback_image)
        rospy.Subscriber(cmd_vel_topic, Twist, self.callback_action)
        if reset_environment:
            self.coord_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=2)

    def new_traj(self):
        if self.gather:
            print('Transitions in last Traj',len(self.states_buffer[-1]))
            if self.always == True:
                self.states_buffer.append([])
                self.actions_buffer.append([])
            else: 
                if len(self.states_buffer) < self.buffer_size :
                    self.states_buffer.append([])
                    self.actions_buffer.append([])
                else: 
                    self.gather = False 
                    
    
    
    def callback_action(self, msg):
        if self.gather:
            self.action.append((msg.linear.x, msg.angular.z))


    def callback_image(self, msg):
        # Convert ROS Image to OpenCV format
        if self.gather:

            image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transforms.ToTensor()(image)
            if self.im_preproc:
                image = self.preprocess(image)
            self.states_buffer[-1].append(image)
            self.actions_buffer[-1].append(self.action[0])
            if len(self.states_buffer[-1])==self.num_transitions:
                self.new_traj()
                if self.reset_environment:
                    rospy.wait_for_service('/gazebo/reset_world')
                    reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
                    reset_world()
                    self.set_cube_coord()


 




    def new_data(self):
        self.states_buffer = deque([[]], maxlen=self.buffer_size)
        self.actions_buffer = deque([[]], maxlen=self.buffer_size)
        self.action = deque([(0, 0)], maxlen=1)
        self.gather = True

    def set_cube_coord(self):
        theta_cube = random.uniform(0, 2 * math.pi)
        theta_sphere = theta_cube + math.pi
        x_cube = 8 * math.cos(theta_cube)
        y_cube = 8 * math.sin(theta_cube)
        x_sphere = 8 * math.cos(theta_sphere)
        y_sphere = 8 * math.sin(theta_sphere)
        cube_pose = ModelState()
        cube_pose.model_name = "box_model"
        sphere_pose = ModelState()
        sphere_pose.model_name = "sphere_model"
        cube_pose.pose.position.x = x_cube
        cube_pose.pose.position.y = y_cube
        cube_pose.pose.position.z = 0.5 
        sphere_pose.pose.position.x = x_sphere
        sphere_pose.pose.position.y = y_sphere
        sphere_pose.pose.position.z = 0.5
        self.coord_pub.publish(sphere_pose)
        self.coord_pub.publish(cube_pose)




def rospy_thread():
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass


def main_thread():
    while traj_buffer.gather == True:
        time.sleep(1)
    #print(traj_buffer.states_buffer[-1].shape)
    #print(traj_buffer.states_buffer[-1][0])


if __name__ == '__main__':
    # Create the ImageBuffer instance
    traj_buffer = TrajectoryBuffer(
        buffer_size=2,  
        im_resolution=(640,480),  
        num_transitions=10, 
        always = True,
        image_topic= '/camera/rgb/image_raw',
        cmd_vel_topic= '/cmd_vel',
        reset_environment= False
        )

    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=main_thread)
    t1.start()
    t2.start()
    
    t2.join()
    rospy.signal_shutdown('Buffer gather finished')   