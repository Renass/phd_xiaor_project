import torch
import os
import trajectories_gather5
import threading
import rospy
import time
from archive_code_supported.visualBERT_train import CubeClassifier
from geometry_msgs.msg import Twist

'''
Behavioral cloning VisualBERT transformer inference
Actions are resolved as a regression task
hdf Dataset

im1,    mask,   mask, mask, action_tken
im1, im2,   mask,   mask,     action_token
im1,    im2,    im3,   mask,    action_token
im1, im2,    im3,     im4,    action_token

import:
trajectories_gather5, visualBERT_train
'''

def publish_twist(publisher, a):
    twist_msg = Twist()
    twist_msg.linear.x = a[0]
    twist_msg.linear.y = 0.0
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = a[1]
    publisher.publish(twist_msg)


def rospy_thread():
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass


def behav_clon_inference_thread():
    while not rospy.is_shutdown():
        print('inference step starts')
        while traj_buffer.gather == True:
            if len(traj_buffer.states_buffer) > 0:
                if len(traj_buffer.states_buffer[-1]) >  0:
                    start_time = time.time()
                    current_state = traj_buffer.states_buffer[-1][-1].unsqueeze(0).unsqueeze(0).to(device)
                    #print('current state: ', current_state.shape)

                    output = model.forward(current_state, prompt=PROMPT)
                    #print('cnn_output', cnn_output.shape)
                    action = output[0].cpu().detach().numpy()
                    publish_twist(driv_pub, action)
                    #prob = cnn_output[-1].data[1].item()*100
                    #format_prob = "{:.2f}".format(prob)
                    
                    end_time = time.time()
                    print('one_move time :', end_time - start_time)


BUFFER_SIZE = 1
IM_RESOLUTION = (224, 224)
SEQUENCE_LENGTH = 300
PROMPT = "Go not to the cube"


if __name__ == '__main__':


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Cuda Device: ',device, torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
    print('Current device: ',device)


    model = CubeClassifier(device=device)
    model = model.to(device)  
    model.eval()

    cnn_weights_file = 'visualBERT.pt'
    if os.path.isfile(cnn_weights_file):
        model.load_state_dict(torch.load(cnn_weights_file))
        print('weights loaded from file.')

    traj_buffer = trajectories_gather5.TrajectoryBuffer(
        buffer_size=BUFFER_SIZE,  im_resolution=IM_RESOLUTION, 
        num_transitions=SEQUENCE_LENGTH, always=True)
    
    driv_pub = rospy.Publisher('robot_base_velocity_controller/cmd_vel', Twist, queue_size=1)

    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=behav_clon_inference_thread)
    t1.start()
    print('Traj gather starts')
    t2.start()
    print('Cube Classifier inference starts')