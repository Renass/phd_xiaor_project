from renas_train4 import Renas, StateActionPromptDataset
import torch
import trajectories_gather6
import threading
import rospy
import time
import numpy as np
from geometry_msgs.msg import PoseStamped

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

#whatever the size is - the only current episode would be predicted action
BUFFER_SIZE = 1
ACTION_ROSTOPIC = '/move_base_simple/goal'


def rospy_thread():
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except:
            pass

def publish_pose(publisher, action):
    goal_msg = PoseStamped()
    goal_msg.header.frame_id = 'map'
    goal_msg.pose.position.x = action[0]
    goal_msg.pose.position.y = action[1]
    goal_msg.pose.position.z = 0

    goal_msg.pose.orientation.x = 0
    goal_msg.pose.orientation.y = 0
    goal_msg.pose.orientation.z = action[2]
    goal_msg.pose.orientation.w = action[3]

    publisher.publish(goal_msg)

def behav_clon_inference_thread():
    while not rospy.is_shutdown():
        if traj_buffer.waiting == 'action':
            start_time = time.time()
            im = (torch.from_numpy(np.stack(traj_buffer.states_buffer[-1], axis=0))/255.0).type(torch.float32).unsqueeze(0) 
            map = (torch.from_numpy(np.stack(traj_buffer.map_buffer[-1], axis=0))/100.0).type(torch.float32).unsqueeze(0)
            costmap = (torch.from_numpy(np.stack(traj_buffer.costmap_buffer[-1], axis=0))/100.0).type(torch.float32).unsqueeze(0)
            mapinfo = traj_buffer.map_info
            mapinfo = torch.tensor([
                mapinfo['resolution'],
                mapinfo['width'],
                mapinfo['height'],
                mapinfo['origin']['position']['x'],
                mapinfo['origin']['position']['y'],
                mapinfo['origin']['position']['z'],
                mapinfo['origin']['orientation']['x'],
                mapinfo['origin']['orientation']['y'],
                mapinfo['origin']['orientation']['z'],
                mapinfo['origin']['orientation']['w'] 
            ], dtype=torch.float32).unsqueeze(0)
            pose = torch.from_numpy(np.stack(traj_buffer.pose_buffer[-1], axis=0)).type(torch.float32).unsqueeze(0)
            action = torch.zeros((1,1,4)) 
            if len(traj_buffer.actions_buffer[-1])>0:
                action = torch.cat((torch.from_numpy(np.stack(traj_buffer.actions_buffer[-1], axis=0)), action[0]), dim=0).type(torch.float32).unsqueeze(0)
            prompt = traj_buffer.task_buffer[-1]


            output = model((im, map, costmap, mapinfo, pose, action, prompt))[-1][-1]
            

            #print(im.shape)            
            #print(map.shape)
            #print(costmap.shape)
            #print(mapinfo.shape)
            #print(pose.shape)
            #print(action.shape)
            if output[2]**2+output[3]**2>0.5: 
                publish_pose(driv_pub, output)
                print('Model published action')
            else:
                print('Model wants to end the episode')    
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

    driv_pub = rospy.Publisher(ACTION_ROSTOPIC, PoseStamped, queue_size=1)

    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=behav_clon_inference_thread)
    t1.start()
    print('Traj gather starts')
    t2.start()
    print('inference starts')