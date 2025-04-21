#turn off warnings
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import sys
#import in python from 1 level parental directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from renas10_train_flava import Renas10forTrain, draw_an_arrow_on_the_map, action2token_vocab 
import trajectories_gather6

import torch
import threading
import rospy
import time
import numpy as np
from geometry_msgs.msg import PoseStamped
import h5py
import torch.nn.functional as F
import matplotlib.pyplot as plt
from diagnostic_msgs.msg import KeyValue
import json



'''
Behavioral cloning camera-lidar INFERENCE for Renas MODEL 10
*IM PRETRAIN WITH ARROW LIKE IN RENAS6 MODEL

File work:
    input:
        action_annotation.h5 - image descriptions of action options
        action_annotation_tasks.txt - prompt annotations of action options 
        action_annotation_mapinfo.json
MODEL 10:
    Behavioral cloning Renas  transformer camera-lidar
    1. TEXT-Image camera or (camera+map concatenation) ENCODER using ViLT 
    2. NO TEXT GENERATION 
    
    3. (im_prompt)-(action) history-aware causal driving Transformer GPT
    Loss: cross-attention metrics going to CrossEntropyLoss 
    Similarity metric: First half of cross-attention

DATA:
    1. Behavioral cloning correct demonstrations (state-action episodes) 

    State: (image) or (im-map concatenation), prompt 

    Actions in ros: position(x,y) orientation quternions (z, w)
    Actions for model are explored (im-prompt description) and set as tokens vocabulary

    2. Actions annotations
    (Im) or (Im-map), prompt

new task:
rostopic pub /task diagnostic_msgs/KeyValue "{key: 'new_task', value: 'go left'}"
end task:
rostopic pub /task diagnostic_msgs/KeyValue "{key: 'end_task', value: 'done'}"
'''

IMAGE_TOPIC = '/camera/rgb/image_raw'
#IMAGE_TOPIC = '/image_raw'

#For SLAM:
#MAP_SERVICE = '/dynamic_map'
#For AMCL:
MAP_SERVICE = '/static_map'
ACTION_ROSTOPIC = '/move_base_simple/goal'

#whatever the size is - the only current episode would be predicted action
BUFFER_SIZE = 1

#ACTION_VOCAB = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/2poses/poses/poses_2024-04-25_15-00-52.h5'
ACTION_VOCAB = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/real/2A724_may/poses/poses_2024-05-04_18-10-20.h5'
#ACTION_ANNOTATION = '/move_base_simple/goal'

# Action options transfered to embeddings (files end with action_vocab.h5)
#POSES = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/poses/poses_2024-04-25_15-00-52_action_vocab.h5'

WEIGHTS_DIR = '/home/renas/pythonprogv2/phd_xiaor_project/weights/'
LOAD_WEIGHTS = 'renas10_vilt6.pt'

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
            torch.cuda.synchronize()
            start_time = time.time()
            
            act_vocab_token = model.annot_forward(act_vocab_im, act_vocab_prompt, act_vocab_map) 


            im = (torch.from_numpy(np.stack(traj_buffer.states_buffer[-1], axis=0))).type(torch.float32).permute(0, 3, 1, 2)
            im = F.interpolate(im, size=(112,224), mode='bilinear', align_corners=False)
            im = im/255.0

            pose = traj_buffer.pose_buffer[-1]

            map = np.stack(traj_buffer.map_buffer[-1], axis=0)/100
            map = draw_an_arrow_on_the_map(map, mapinfo, pose)
            map = torch.from_numpy(map).float()
            map = F.interpolate(map, size=(112,224), mode='bilinear', align_corners=False)
            #plt.imshow(im[0][0].numpy().transpose(1,2,0))
            #plt.show()
            if len(traj_buffer.actions_buffer[-1])>0: 
                action = np.stack(traj_buffer.actions_buffer[-1], axis=0)
                action = torch.from_numpy(action)
                #EOS token
                action = torch.cat((action, torch.ones((1,4))), dim=0)
            else:
                action = torch.ones((1,4))

            if 'new_task' in traj_buffer.task_buffer[-1]:
                prompt = [traj_buffer.task_buffer[-1]["new_task"]]
            #a_label = action2label_vocab(action[0], action_vocab_action)
            #action = action2token_vocab(action[0], action_vocab_token, action_vocab_action) 
            
            output = model((im.unsqueeze(0), prompt, action.unsqueeze(0), None, map.unsqueeze(0)), act_vocab_token, act_vocab_coords)
            output = output.squeeze(0)
            #print('labels: ', a_label)
            #print('raw digits of actions to choose:')
            #print(output)
            _, output = torch.max(output[-1], 0)
            output = act_vocab_coords[output.cpu()]
            if output[2]< 0.9 or output[3]<0.9: 
                #publish_pose(driv_pub, output)
                #publish_pose(driv_pub, [4.1, -3.5, -0.4, 0.9])

                print('Model published action')
            else:
                print('Model wants to end the episode')
                task_msg = KeyValue()
                task_msg.key = 'end_task'
                task_msg.value = 'reason: success, end by model'
                #task_pub.publish(task_msg)
                #publish_pose(driv_pub, [4.1, -3.5, -0.4, 0.9])
            torch.cuda.synchronize()    
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

    #load model
    model = Renas10forTrain(device).to(device)
    model.eval()

    
    if os.path.isfile(os.path.join(WEIGHTS_DIR, LOAD_WEIGHTS)):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(os.path.join(WEIGHTS_DIR, LOAD_WEIGHTS))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        del model_dict, pretrained_dict
        print('weights loaded from file.')
    else:
        print("weights wasn't loaded")

    #load action vocab
    act_vocab_prompt = []
    act_vocab_map = []
    act_vocab_im = []
    act_vocab_coords = []
    annot_prompt_filename = ACTION_VOCAB[:-3]+'_tasks.txt'
    with open(annot_prompt_filename, 'r') as file:
        for p in file:
            act_vocab_prompt.append(p.strip())
    print(act_vocab_prompt)

    annot_mapinfo_filename = f"{os.path.splitext(ACTION_VOCAB)[0]}_mapinfo.json"
    with open(annot_mapinfo_filename, 'r') as file:
        annot_mapinfo = json.load(file)
    mapinfo = annot_mapinfo
    with h5py.File(ACTION_VOCAB, 'r') as annot_hdf:
        im_group = annot_hdf['states']
        map_group =annot_hdf['maps']
        pose_group = annot_hdf['pose']
        action_group = annot_hdf['actions']
        num_annots = len(im_group)
        print('ACTION VOCAB contains options: ', num_annots)
        for i in range(num_annots+1):
            if i<num_annots:
                #For annons except EOS token
                annot = 'data_'+str(i)
                pose_i = pose_group[annot][:]
                map_i = map_group[annot][:]/100
                map_i = draw_an_arrow_on_the_map(map_i, annot_mapinfo, pose_i)
                map_i = torch.from_numpy(map_i).float()
                map_i = F.interpolate(map_i, size=(112,224), mode='bilinear', align_corners=False)
                act_vocab_map.append(map_i)
                im_i = torch.from_numpy(im_group[annot][0]).float().permute(2,0,1).unsqueeze(0)
                im_i = F.interpolate(im_i, size=(112,224), mode='bilinear', align_corners=False).squeeze(0)
                act_vocab_im.append(im_i//255.0)   
                act_vocab_coords.append(torch.from_numpy(action_group[annot][0]))
            else:
                #For EOS token
                #act_vocab_im.append(torch.ones_like(act_vocab_im[0]))
                act_vocab_coords.append(torch.ones_like(act_vocab_coords[0]))
        act_vocab_coords = torch.stack(act_vocab_coords, dim=0)

    traj_buffer = trajectories_gather6.TrajectoryBuffer(
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