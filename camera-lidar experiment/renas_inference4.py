from renas_train4 import Renas
import torch
import trajectories_gather6
import threading
import rospy
import time
import numpy as np

'''
Behavioral cloning Renas  transformer camera-lidar INFERENCE

State: im, map, costmap, pose, mapinfo, prompt

Actions in ros: position(x,y) orientation quternions (z, w)

1. TEXT-Image encoding using ViLT (trainable) 
2. >Text-Image token + lidar map, costmap, pose self-attention transformer 
3. (State)-(action) causal Transformer GPT 
'''


IMAGE_TOPIC = '/image_raw'
LOAD_WEIGHTS = '/home/renas/pythonprogv2/phd_xiaor_project/weights/renas4.pt'

#For SLAM:
#MAP_SERVICE = '/dynamic_map'
#For AMCL:
MAP_SERVICE = '/static_map'

BUFFER_SIZE = 1


def rospy_thread():
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass

def behav_clon_inference_thread():
    while not rospy.is_shutdown():
        #print('inference step starts')
        #while True:
            #time.sleep(1)
        if traj_buffer.waiting == 'action':
            start_time = time.time()
            print('here')
            #Following data collection on writer program and
            #following data preprocessing on renas_train4.py
            im = torch.from_numpy(np.stack(traj_buffer.states_buffer[-1], axis=0)).float()/255.0 
            if len(traj_buffer.actions_buffer[-1])>0:
                print('actions >0')
                action = torch.from_numpy(np.stack(traj_buffer.actions_buffer[-1], axis=0))
            else:
                print('actions<0')
            print(action.shape)

            print('one_move time :', time.time() - start_time)
            time.sleep(1)









if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Cuda Device: ',device, torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
    print('Current device: ',device)


    model = Renas(device=device)
    model = model.to(device)  
    model.eval()

    model.load_state_dict(torch.load(LOAD_WEIGHTS))
    print('weights loaded from file.')

    traj_buffer = trajectories_gather6.TrajectoryBuffer(
    image_topic= IMAGE_TOPIC,
    map_service= MAP_SERVICE,
    buffer_size= BUFFER_SIZE,
    always= True
    )

    #driv_pub = rospy.Publisher(CMD_PUBLISH_TOPIC, Twist, queue_size=1)

    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=behav_clon_inference_thread)
    t1.start()
    print('Traj gather starts')
    t2.start()
    print('inference starts')