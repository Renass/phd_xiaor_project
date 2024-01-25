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

'''
Node of gathering state-action-return trajectories
work with: reset_env.py, reward_publisher.py
(Used to be imported) 
'''



class TrajectoryBuffer:
    def __init__(self, buffer_size=10, im_amount=5, im_resolution = (640,480), im_channels = 3, always = True):
        self.always = always
        self.im_resolution = im_resolution
        self.im_channels = im_channels
        self.traj = deque([[]], maxlen=buffer_size)
        self.gather = True
        self.buffer_size = buffer_size
        self.im_amount = im_amount
        self.images_buffer = deque(maxlen=im_amount)
        self.reward = deque(maxlen=1)
        self.action = deque([(0, 0)], maxlen=1)
        self.tensor_images = None
        self.bridge = CvBridge()

        # Initialize the ROS node and subscribe to the image topic
        rospy.init_node('multimodal_inference_node', anonymous=True)
        rospy.Subscriber('/image_raw', Image, self.callback_image)
        rospy.Subscriber('gazebo/rl_reward', String, self.callback_reward)
        rospy.Subscriber('robot_base_velocity_controller/cmd_vel', Twist, self.callback_action)
        rospy.Subscriber('reset_signal', String, self.new_traj)

    def new_traj(self, msg):
        if self.gather:
            print('Transitions in last Traj',len(self.traj[-1]))
            if self.always == True:
                self.traj.append([])
            else: 
                if len(self.traj) < self.buffer_size :
                    self.traj.append([])
                else: 
                    self.gather = False 
                    
    
    def callback_reward(self, msg):
        if self.gather:
            self.reward.append(msg.data)

    
    def callback_action(self, msg):
        if self.gather:
            self.action.append([msg.linear.x, msg.angular.z])


    def callback_image(self, msg):
        # Convert ROS Image to OpenCV format
        if self.gather:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.images_buffer.append(cv_image)

            # Check if we have received the desired number of images
            if len(self.images_buffer) == self.im_amount:
                #self.gather = False
                self.get_last_images_as_tensor()
                self.traj[-1].append([self.tensor_images, self.action[0], self.reward])
                #os.system('clear')
                #print(self.traj)


 

    def get_last_images_as_tensor(self):
        
        # Get the last images from the buffer as a PyTorch tensor
        valid_images = [img for img in self.images_buffer if img is not None]
        if len(valid_images) == 0:
            return None
        
        stacked_images = np.stack(valid_images)
        self.tensor_images = torch.tensor(stacked_images.transpose((0, 3, 1, 2)), dtype=torch.float32)
        self.tensor_images = torch.reshape(self.tensor_images,(self.im_channels,self.im_resolution[1]*self.im_amount,self.im_resolution[0]) )
        #self.tensor_images = torch.cat((self.tensor_images,), dim=1)
        #rospy.signal_shutdown("Received the desired number of images")


    def new_data(self):
        self.traj = deque([[]], maxlen=self.buffer_size)
        self.images_buffer = deque(maxlen=self.im_amount)
        self.tensor_images = None
        self.action = deque([(0, 0)], maxlen=1)
        self.reward = deque(maxlen=1)
        self.gather = True
    
if __name__ == '__main__':
    # Create the ImageBuffer instance
    traj_buffer = TrajectoryBuffer(buffer_size=2, im_resolution=(224,224), always = True)
    #traj_buffer.fullfill_trajectory()
    #while len(traj_buffer.images_buffer)<traj_buffer.im_amount:
        #time.sleep(0.1)

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass
    