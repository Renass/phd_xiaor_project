from renas6_train import Renas, StateActionPromptDataset
import torch
import trajectories_gather7
import threading
import rospy
import time
import numpy as np
from geometry_msgs.msg import PoseStamped
import h5py
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diagnostic_msgs.msg import KeyValue

'''
Behavioral cloning Renas  transformer camera-lidar INFERENCE

State: im-map concatenation (reworked h5), prompt 
states organized as sequences - episodes

Actions in ros: position(x,y) orientation quternions (z, w)
Actions for model are explored (im-prompt description) and set as tokens vocabulary

1. TEXT-Image(camera+map concatenation) encoding using ViLT (trainable) 
2. (im_prompt)-(action) causal Transformer GPT 

new task:
rostopic pub /task diagnostic_msgs/KeyValue "{key: 'new_task', value: 'go left'}"
end task:
rostopic pub /task diagnostic_msgs/KeyValue "{key: 'end_task', value: 'done'}"
'''

IMAGE_TOPIC = '/camera/rgb/image_raw'
#IMAGE_TOPIC = '/image_raw'

LOAD_WEIGHTS = '/home/renas/pythonprogv2/phd_xiaor_project/weights/early_renas6.pt'

#For SLAM:
#MAP_SERVICE = '/dynamic_map'
#For AMCL:
MAP_SERVICE = '/static_map'

#whatever the size is - the only current episode would be predicted action
BUFFER_SIZE = 1
ACTION_ROSTOPIC = '/move_base_simple/goal'

# Action options transfered to embeddings (files end with action_vocab.h5)
POSES = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/real/poses/poses_2024-05-04_18-10-20_action_vocab.h5'

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
            im = (torch.from_numpy(np.stack(traj_buffer.states_buffer[-1], axis=0))).type(torch.float32).unsqueeze(0)
            #plt.imshow(im[0][0].numpy().transpose(1,2,0))
            #plt.show() 
            action = torch.ones((1,1,4)) 
            if len(traj_buffer.actions_buffer[-1])>0:
                action = torch.cat((torch.from_numpy(np.stack(traj_buffer.actions_buffer[-1], axis=0)), action[0]), dim=0).type(torch.float32).unsqueeze(0)
            if 'new_task' in traj_buffer.task_buffer[-1]:
                prompt = [traj_buffer.task_buffer[-1]["new_task"]]
            #a_label = action2label_vocab(action[0], action_vocab_action)
            action = action2token_vocab(action[0], action_vocab_token, action_vocab_action) 
            
            output = model((im, action, action, prompt), action_vocab_token)[-1][-1]
            
            print(output)
            #print('labels: ', a_label)
            _, output = torch.max(output, 0)
            output = action_vocab_action[output.cpu()]

            if output[2]< 0.9 or output[3]<0.9: 
                publish_pose(driv_pub, output)
                print('Model published action')
            else:
                print('Model wants to end the episode')
                task_msg = KeyValue()
                task_msg.key = 'end_task'
                task_msg.value = 'reason: success, end by model'
                task_pub.publish(task_msg)    
            print('one_move time :', time.time() - start_time)
            time.sleep(1)

def action2label_vocab(action, action_vocab_action):
    action = action.unsqueeze(1)
    action_vocab_action = action_vocab_action.unsqueeze(0) 
    similarity_scores = F.cosine_similarity(action, action_vocab_action, dim=2)
    max_values, max_indices = torch.max(similarity_scores, dim=1)
    return max_indices

def action2token_vocab(action, action_vocab_token, action_vocab_action):
    action = action.unsqueeze(1)
    action_vocab_action = action_vocab_action.unsqueeze(0) 
    similarity_scores = F.cosine_similarity(action, action_vocab_action, dim=2)
    max_values, max_indices = torch.max(similarity_scores, dim=1)
    selected_tokens = [action_vocab_token[idx] for idx in max_indices]
    selected_tokens = torch.stack(selected_tokens, dim=0)
    return selected_tokens

if __name__ == '__main__':
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device('cuda:0')
            device_i = torch.device(f'cuda:{i}')
            print(f'Cuda Device {i}: ', device_i, torch.cuda.get_device_name(i))
    else:
        print('No CUDA devices available')
        device = torch.device('cpu')
    print('Current device: ',device)


    model = Renas(device=device)
    model = model.to(device)  
    model.eval()

    model.load_state_dict(torch.load(LOAD_WEIGHTS))
    print('weights loaded from file.')


    action_vocab_token = []
    action_vocab_action = []
    with h5py.File(POSES, 'r') as hdf2:
        num_poses = len(hdf2['tokens'])
        for i in range(num_poses):
            action_vocab_token.append(torch.from_numpy(hdf2['tokens']['data_'+str(i)][:]))
            action_vocab_action.append(torch.from_numpy(hdf2['actions']['data_'+str(i)][:])[0])
        action_vocab_token = torch.stack(action_vocab_token, dim=0)
        #additional end_token of ones
        action_vocab_token = torch.cat((action_vocab_token, torch.ones((1, 768))))

        action_vocab_action = torch.stack(action_vocab_action, dim=0)
        #additional end_token of ones
        action_vocab_action = torch.cat((action_vocab_action, torch.ones((1, 4))))  

    traj_buffer = trajectories_gather7.TrajectoryBuffer(
    image_topic= IMAGE_TOPIC,
    map_service= MAP_SERVICE,
    buffer_size= BUFFER_SIZE,
    always= True
    )

    driv_pub = rospy.Publisher(ACTION_ROSTOPIC, PoseStamped, queue_size=1)
    task_pub = rospy.Publisher('/task', KeyValue, queue_size=1)
    
    t1 = threading.Thread(target=rospy_thread)
    t2 = threading.Thread(target=behav_clon_inference_thread)
    t1.start()
    print('Traj gather starts')
    t2.start()
    print('inference starts')