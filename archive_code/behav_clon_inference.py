import torch
import behav_clon_train
from behav_clon_train import (STATE_DIM, ENCODER_MODEL_NAME, DECODER_MODEL_NAME, 
    DECODER_STATE_REPRES, preprocess)
import rospy
import threading
import trajectories_gather4
from geometry_msgs.msg import Twist
import time
from torchvision import transforms
import numpy as np

'''
Behav clon inference
imports: behav_clon_train.py, trajectories_gather4 
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
            if len(traj_buffer.traj) > 0:
                if len(traj_buffer.traj[-1]) >  0:
                    start_time = time.time()
                    current_trajectory = traj_buffer.traj[-1]
                    states_list = []
                    actions_list = []
                    for transition in current_trajectory:
                        state = transition[0]
                        state = preprocess(state)

                        states_list.append(state)
                        actions_list.append(transition[1])
                    states_tensor = torch.stack(states_list, dim=0).unsqueeze(0)
                    actions_tensor = torch.tensor(actions_list).unsqueeze(0)
                    states_tensor = states_tensor.to(device)
                    actions_tensor = actions_tensor.to(device)
                    encoder_output = encoder_model.forward(states_tensor)
                    decoder_output = decoder_model.forward(encoder_output.last_hidden_state, actions_tensor)
                    action = decoder_output[0][-1].cpu().detach().numpy()
                    #if np.abs(action[0])>np.abs(action[1]):
                        #publish_twist(driv_pub, [2, 0])
                        #print('Drive signal: [2, 0]')
                    #else:
                        #if action[1]>0:
                            #publish_twist(driv_pub, [0,2])
                            #print('Drive signal: [0, 2]')
                        #else:
                            #publish_twist(driv_pub, [0, -2])
                            #print('Drive signal: [0, -2]')
                    publish_twist(driv_pub, action)
                    #print('Decoder oputput: ', decoder_output[0][-1].data)
                    print('Decoder oputput: ', action)
                    end_time = time.time()
                    #print('one_move time :', end_time - start_time)

BUFFER_SIZE = 1
IM_AMOUNT = 1
IM_RESOLUTION = (224,224)
IM_CHANNELS = 3 
SEQUENCE_LENGTH  = 200


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Cuda Device: ',device, torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
    print('Current device: ',device)

    encoder_model = behav_clon_train.CustomEncoder(num_dim=STATE_DIM, model_name = ENCODER_MODEL_NAME)
    encoder_model = encoder_model.to(device)
    encoder_model.eval()

    decoder_model = behav_clon_train.CustomDecoder(model_name=DECODER_MODEL_NAME, state_representation_size=DECODER_STATE_REPRES, action_dim=2)
    decoder_model = decoder_model.to(device)
    decoder_model.eval()

    encoder_weights_file = 'encoder_model_final.pt'
    decoder_weights_file = 'decoder_model_final.pt'
    encoder_model.load_state_dict(torch.load(encoder_weights_file))
    decoder_model.load_state_dict(torch.load(decoder_weights_file))
    print('weights loaded from file.')


    traj_buffer = trajectories_gather4.TrajectoryBuffer(
        buffer_size=BUFFER_SIZE, im_amount=IM_AMOUNT, im_resolution=IM_RESOLUTION, 
        im_channels=IM_CHANNELS, num_transitions=SEQUENCE_LENGTH, always=True)

    driv_pub = rospy.Publisher('robot_base_velocity_controller/cmd_vel', Twist, queue_size=1)


    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=behav_clon_inference_thread)
    t1.start()
    print('Traj gather starts')
    t2.start()
    print('Behavioral cloning inference starts')