import torch
import h5py
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import OpenAIGPTConfig, OpenAIGPTModel
import math
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.utils.tensorboard import SummaryWriter
import os
import time
import shutil
import json
import cv2
from scipy.spatial.transform import Rotation as R
#from transformers import ViltProcessor, ViltModel
from transformers import VisualBertModel, ViltProcessor



'''
*ERROR VisualBERT requires manual image processing with fastRCNN
TRAIN LOOP for Renas MODEL 10 with multimodal encoder Included in training
*ALL steps in one file - no prior preprocessing steps
*IM PRETRAIN WITH ARROW LIKE IN RENAS6 MODEL

File work:
    input:
        tsa_combined.h5 (demonstrations dataset)
        tsa_combined_tasks.txt (demonstrations dataset task prompts)
        action_annotation.h5 - image descriptions of action options
        action_annotation_tasks.txt - prompt annotations of action options 
   
MODEL 10:
    Behavioral cloning Renas  transformer camera-lidar
    1. TEXT-Image camera or (camera+map concatenation) ENCODER using VisualBERT 
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
'''
#Main real dataset
DATASET = '/data/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/real/2A724_may/tsa_combined.h5'
ACTION_ANNOTATION = '/data/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/real/poses/poses_2024-05-04_18-10-20.h5'

#Sim dataset target 100%
#DATASET = '/data/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/tsa_combined.h5'
#ACTION_ANNOTATION = '/data/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/poses/poses_2024-04-25_15-00-52.h5'

DEVICE = 'cuda:0'

LR = 10e-7
LR_WARMUP_EPOCHS = 5 
LR_DECAY_EPOCHS = 100
UPDATE_ANNOT_RATE = 1

TEST_PART = 0.2
BATCH_SIZE = 1
CHECKPOINT_INTERVAL = 25

WEIGHTS_DIR = '/data/renas/pythonprogv2/phd_xiaor_project/weights'
LOAD_WEIGHTS = 'none'
SAVE_WEIGHTS = 'none'

###
#CLASSES
###

class EncodingVector(torch.nn.Module):
    def __init__(self, d_model):
        super(EncodingVector, self).__init__()
        self.modality_vector = torch.nn.Parameter(torch.randn(d_model))
    def forward(self, x):
        return x + self.modality_vector.unsqueeze(0).unsqueeze(0)

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MyDataset(Dataset):
    def __init__(self, im, prompt, actions, a_labels, map):
        self.im = im
        self.prompt = prompt
        self.actions = actions
        self.a_labels = a_labels
        self.map = map
    def __len__(self):
        return len(self.im)
    def __getitem__(self, idx):
        im = self.im[idx]
        prompt = self.prompt[idx]
        action = self.actions[idx]
        a_label = self.a_labels[idx]
        map = self.map[idx]
        return im, prompt, action, a_label, map
    
class Renas10forTrain(torch.nn.Module):
    def __init__(self, device):
        super(Renas10forTrain, self).__init__()
        self.device = device
        #set d_model manually in exceptional cases
        #self.d_model = 2048
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.processor.current_processor.do_rescale = False
        self.processor.current_processor.do_resize = False
        self.processor.current_processor.do_normalize = False

        self.vlm_model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa")
        self.d_model = self.vlm_model.config.hidden_size
        for param in self.vlm_model.parameters():
            param.requires_grad = True  

        self.pos_enc = PositionalEncoding(d_model=self.d_model)
        self.im_prompt_enc_vector = EncodingVector(d_model=self.d_model)
        self.actions_enc_vector = EncodingVector(d_model=self.d_model)
        
        self.gpt_config = OpenAIGPTConfig(vocab_size=0, n_positions=200, n_embd=self.d_model, n_layer=7, n_head=32)
        self.gpt_model = OpenAIGPTModel(self.gpt_config)

        #Weights for final cross-attention multiple choice
        self.q_weights = torch.nn.Linear(self.d_model, self.d_model)
        self.k_weights = torch.nn.Linear(self.d_model, self.d_model)

    def annot_forward(self, act_vocab_im, act_vocab_prompt, act_vocab_map):
        with torch.no_grad():
            #VLM encoder
            #Work with action annotation
            act_vocab_token = []
            for i, im_i in enumerate(act_vocab_im):
                im_i = torch.cat((im_i, act_vocab_map[i][0]), dim=1)
                inputs = self.processor(images=im_i, text= act_vocab_prompt[i], return_tensors="pt")
                #inputs = self.processor(images=[act_vocab_map[i][j] for j in range(1)], text=[act_vocab_prompt[i]], return_tensors="pt")

                #im_i = torch.cat((inputs['pixel_values'], inputs2['pixel_values']), dim=3)
                #inputs = self.processor(images=[im_i[j] for j in range(1)], text=[act_vocab_prompt[i]], return_tensors="pt")
                

                inputs['visual_embeds'] = inputs.pop('pixel_values')
                inputs.pop('pixel_values', None)
                inputs.pop('pixel_mask', None)
                inputs['visual_attention_mask'] = torch.ones(inputs['visual_embeds'].shape[:-1])
                print('here')
                print(inputs.keys())
                print(inputs['input_ids'].shape)
                print(inputs['attention_mask'].shape)
                print(inputs['visual_embeds'].shape)
                print(inputs['visual_attention_mask'].shape)

                inputs = {key: val.to(device) for key, val in inputs.items()}
                outputs = self.vlm_model.forward(**inputs, return_dict=True)
                #print('Action annotation token shape:', outputs.pooler_output.shape)
                act_vocab_token.append(outputs.pooler_output)     
            #For EOS token
            act_vocab_token.append(torch.ones_like(act_vocab_token[0]))
            act_vocab_token = torch.cat(act_vocab_token, dim=0)
            return act_vocab_token

    def forward(self, batch, act_vocab_token):
        #VLM encoder
        #work with dataset
        im, prompt, action, _, map = batch
        state = []
        for i, im_i in enumerate(im):
            episode_len = im_i.shape[0]
            im_i = torch.cat((im_i, map[i]), dim=2)
            inputs = self.processor(images=im_i, text=[prompt[i]]*episode_len, return_tensors="pt")
            #inputs = self.processor(images=[map[i][j] for j in range(episode_len)], text=[prompt[i]]*episode_len, return_tensors="pt")
            
            #im_i = torch.cat((inputs['pixel_values'], inputs2['pixel_values']), dim=3)
            
            #inputs = self.processor(images=[im_i[j] for j in range(episode_len)], text=[prompt[i]]*episode_len, return_tensors="pt")
            #inputs['pixel_values'] = torch.cat((inputs['pixel_values'], inputs2['pixel_values']), dim=3)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            outputs = self.vlm_model.forward(**inputs, return_dict=True)
            #print('states shape:', outputs.pooler_output.shape)
            state.append(outputs.pooler_output)
        state = torch.stack(state, dim=0)

        
        action = action2token_vocab(action, act_vocab_coords, act_vocab_token)
              



        #state-action-gpt part
        state = self.im_prompt_enc_vector(state)
        action = self.actions_enc_vector(action)
        state = self.pos_enc(state)
        action = self.pos_enc(action)
        
        batch_size, seq_len, _ = state.shape
        # 2 types of data for gpt
        tokens = torch.zeros(batch_size, seq_len*2, self.d_model, device=self.device)
        tokens[:, 0::2, :] = state
        tokens[:, 1::2, :] = action

        tokens = self.gpt_model(inputs_embeds = tokens).last_hidden_state
        tokens = tokens[:, 0::2, :]
        tokens = self.q_weights(tokens)
        an = self.k_weights(act_vocab_token)
        attention_scores = torch.matmul(tokens, an.unsqueeze(0).transpose(1,2))
        return attention_scores

###
#Functions
###

def action2token_vocab(action, act_vocab_coords, act_vocab_tokens):
    '''
    Convert action on coordinates (quaternions) shaped:
    [seq_length, 4] or [batch_size, seq_length, 4]
    to action in encoder representation (batch_size, seq_length, d_model) 
    '''
    #print('check correctness')
    #print(action.shape)
    #print(act_vocab_coords.shape)
    #print(act_vocab_tokens.shape)

    act_vocab_coords = act_vocab_coords.unsqueeze(0)

    if action.dim() == 2:
        action = action.unsqueeze(0)
    batch_size = action.shape[0] 
    
    # Compute cosine similarity (batch_size, seq_length, vocab_size)
    similarity_scores = F.cosine_similarity(action.unsqueeze(2), act_vocab_coords.unsqueeze(0), dim=-1)

    # Find the max similarity scores and their indices
    max_values, max_indices = torch.max(similarity_scores, dim=-1)
    if torch.min(max_values).item() < 0.999:
        print('Warning: action coordinates not match vocabulary!!!')
    a = []
    aa= []
    for i in range(batch_size):
        for j in max_indices[i]:
            a.append(act_vocab_tokens[j])
        aa.append(torch.stack(a, dim=0))
    aa = torch.stack(aa, dim=0)
    return aa

def padding_collate(batch):
    #num_data_types = len(batch[0])
    new_batch = []
    for i in range(5): # 5 data types: states, prompt, action, a_label, map (skip collate for prompt)
        new_batch.append([item[i] for item in batch])
        if i != 1:
            new_batch[i] = pad_sequence(new_batch[i], batch_first=True, padding_value= 1.0)
    return new_batch


def draw_an_arrow_on_the_map(map, mapinfo, pose):
    '''
    unsqueeze a lidar map to 3 dimensions
    with 1st with map and second with pose arrow
    accept: numpy(batch_size, h, w)
    return: numpy(batch_size, 3, h, w)
    '''
    batch_size,h,w = map.shape
    empty_channel = np.zeros((batch_size, h, w))
    #map = np.expand_dims(map, axis=1)
    map = np.stack((map, empty_channel, empty_channel), axis=1)
    
    
    for i in range(batch_size):
        map_pose = world_to_map(
            (pose[i][0], pose[i][1]), 
            mapinfo['resolution'], 
            (mapinfo['origin']['position']['x'], 
            mapinfo['origin']['position']['y'])
        )
        quaternion = [0, 0, pose[i][2], pose[i][3]]
        rotation = R.from_quat(quaternion)
        yaw = rotation.as_euler('xyz', degrees=False)[2] 
        #_, _, yaw = euler_from_quaternion(quaternion)
        arrow_length = 50
        end_x = int(map_pose[0] + arrow_length * np.cos(yaw))
        end_y = int(map_pose[1] + arrow_length * np.sin(yaw))
        cv2.arrowedLine(map[i, 1, :, :], (map_pose[0], map_pose[1]), (end_x, end_y), 1.0, thickness=5)    
        
        # Visualization using matplotlib
        #plt.imshow(np.flipud(map[i].transpose(1,2,0)))
        #plt.show()
        return map
    
def world_to_map(pose, resolution, origin):
    """
    Convert world coordinates to map pixel coordinates.
    
    :param pose: The pose in world coordinates (x, y).
    :param resolution: The map resolution (meters per pixel).
    :param origin: The origin of the map in world coordinates (x, y).
    :return: The pose in map pixel coordinates.
    """
    map_x =  int((pose[0] - origin[0]) / resolution)
    #map_y = mapinfo['height'] - int((pose[1] - origin[1]) / resolution)
    map_y = int((pose[1] - origin[1]) / resolution)
    return (map_x, map_y)

def action2label_vocab(action, action_vocab_action):
    action = action.unsqueeze(1)
    action_vocab_action = action_vocab_action.unsqueeze(0) 
    similarity_scores = F.cosine_similarity(action, action_vocab_action, dim=2)
    max_values, max_indices = torch.max(similarity_scores, dim=1)
    if torch.min(max_values).item() < 0.999:
        print('Warning: action coordinates not match vocabulary!!!')
    return max_indices

def train_loop(train_dataset, test_dataset, act_vocab_coords, act_vocab_im, act_vocab_prompt, act_vocab_map):
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=padding_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=padding_collate)    
    model = Renas10forTrain(DEVICE).to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=LR_WARMUP_EPOCHS)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=LR_DECAY_EPOCHS, eta_min= LR/10)
    scheduler3 = ConstantLR(optimizer, factor=LR/10, total_iters= 100000)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2, scheduler3], milestones=[LR_WARMUP_EPOCHS, LR_WARMUP_EPOCHS+LR_DECAY_EPOCHS])
    criterion = torch.nn.CrossEntropyLoss()
    ten_board_writer = SummaryWriter()

    if os.path.isfile(os.path.join(WEIGHTS_DIR, LOAD_WEIGHTS)):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(os.path.join(WEIGHTS_DIR, LOAD_WEIGHTS))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        del model_dict, pretrained_dict
        print('weights loaded from file.')

    epoch = 0
    min_loss = 10000
    while True:
        epoch += 1
        total_loss = 0
        test_total_loss = 0
        epoch_train_time_start = time.time()
        optimizer.zero_grad()
        total_accuracy_train = [0, 0]
        total_accuracy_test = [0, 0]
        if (epoch-1)%UPDATE_ANNOT_RATE == 0:
            act_vocab_token = model.annot_forward(act_vocab_im, act_vocab_prompt, act_vocab_map) 
        for i, batch in enumerate(train_dataloader):
            output = model(batch, act_vocab_token)
            # print 1st batch 1st episode labels and predictions
            if i==0:
                print('correct labels: ', batch[3][0])
                print('model output: ')
                softmax_output = F.softmax(output[0], dim=-1)
                formatted_output = [[f"{value.item():.2f}" for value in row] for row in softmax_output]
                for row in formatted_output:
                    print(row)
            output_flat = output.view(-1, output.shape[-1])
            labels_flat = batch[3].to(device).view(-1)
            loss = criterion(output_flat, labels_flat)
            total_loss += loss
            loss.backward()
            _, predicted_classes = torch.max(output_flat, 1)
            #print('Debugging: predicted classes:',predicted_classes)
            total_accuracy_train[0] += (predicted_classes == labels_flat).float().sum()
            total_accuracy_train[1] += labels_flat.size(0) 
        optimizer.step()
        scheduler.step()
        average_loss = total_loss/len(train_dataloader)
        ten_board_writer.add_scalar('Loss', average_loss.item(), epoch)
            
        print('\nEpoch: ', epoch,"  Training Loss:", average_loss.item())
        print("  Training Accuracy:", (total_accuracy_train[0]/total_accuracy_train[1]).item())
        
        #Test part
        with torch.no_grad():
            for batch in test_dataloader:
                output = model(batch, act_vocab_token)
                #output = model(batch, act_vocab_coords, act_vocab_tokens)
                output_flat = output.view(-1, output.shape[-1])
                labels_flat = batch[3].to(device).view(-1)
                test_loss = criterion(output_flat, labels_flat)
                test_total_loss += test_loss
                _, predicted_classes = torch.max(output_flat, 1)
                total_accuracy_test[0] += (predicted_classes == labels_flat).float().sum()
                total_accuracy_test[1] += labels_flat.size(0) 
            
            test_average_loss = test_total_loss/len(test_dataloader)   
            ten_board_writer.add_scalar('Test_Loss', test_average_loss.item(), epoch)
        epoch_train_time_end = time.time()
        print("  Test Accuracy:", (total_accuracy_test[0]/total_accuracy_test[1]).item())
        print('Epoch train time: ',epoch_train_time_end-epoch_train_time_start)
        
        if epoch % CHECKPOINT_INTERVAL == 0:
            if not os.path.exists(WEIGHTS_DIR):
                os.makedirs(WEIGHTS_DIR)
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'temp_'+ SAVE_WEIGHTS))
            shutil.move(os.path.join(WEIGHTS_DIR, 'temp_'+ SAVE_WEIGHTS), os.path.join(WEIGHTS_DIR, SAVE_WEIGHTS))

            if test_average_loss.item()<min_loss:
                torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'temp_'+ 'early_'+ SAVE_WEIGHTS))
                shutil.move(os.path.join(WEIGHTS_DIR, 'temp_'+'early_'+ SAVE_WEIGHTS), os.path.join(WEIGHTS_DIR, 'early_'+ SAVE_WEIGHTS))
                min_loss = test_average_loss.item()
                print('Early stopping with loss', min_loss, 'at the epoch', epoch)
            print('weights saved')












if __name__ == '__main__':
    preprocess_timer_start = time.time()
    if torch.cuda.is_available():
        device = torch.device(DEVICE)
        for i in range(torch.cuda.device_count()):
            device_i = torch.device(f'cuda:{i}')
            print(f'Cuda Device {i}: ', device_i, torch.cuda.get_device_name(i))
    else:
        print('No CUDA devices available')
        device = torch.device('cpu')
    print('Current device: ',device)

    im = []
    map = []
    actions = []
    prompt = []
    act_vocab_prompt = []
    act_vocab_im = []
    act_vocab_coords = []
    act_vocab_map = []
    a_labels = []

    #load prompts
    prompt_filename = DATASET[:-3]+'_tasks.txt'
    with open(prompt_filename, 'r') as file:
        for p in file:
            prompt.append(p.strip())

    annot_prompt_filename = ACTION_ANNOTATION[:-3]+'_tasks.txt'
    with open(annot_prompt_filename, 'r') as file:
        for p in file:
            act_vocab_prompt.append(p.strip())
    print(act_vocab_prompt)


    #load action annotations
    annot_mapinfo_filename = f"{os.path.splitext(ACTION_ANNOTATION)[0]}_mapinfo.json"
    with open(annot_mapinfo_filename, 'r') as file:
        annot_mapinfo = json.load(file)
    with h5py.File(ACTION_ANNOTATION, 'r') as annot_hdf:
        im_group = annot_hdf['states']
        map_group =annot_hdf['maps']
        pose_group = annot_hdf['pose']
        action_group = annot_hdf['actions']
        num_annots = len(im_group)
        print('ACTION ANNOTATION contains options: ', num_annots)
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
                        
    #load demonstrations dataset
    mapinfo_filename = f"{os.path.splitext(DATASET)[0]}_mapinfo.json"
    with open(mapinfo_filename, 'r') as file:
        mapinfo = json.load(file)

    with h5py.File(DATASET, 'r') as hdf:
        im_group = hdf['states']
        map_group =hdf['maps']
        pose_group = hdf['pose']
        action_group = hdf['actions']
        num_episodes = len(im_group)
        print('Dataset contains episodes: ', num_episodes)
        print('Dataset loading and preprocessing...')
        for i in range(num_episodes):
            episode = 'data_'+str(i)
            pose_i = pose_group[episode][:]
            map_i = map_group[episode][:]/100
            map_i = draw_an_arrow_on_the_map(map_i, mapinfo, pose_i)
            map_i = torch.from_numpy(map_i).float()
            map_i = F.interpolate(map_i, size=(112,224), mode='bilinear', align_corners=False)
            map.append(map_i)
            #map_i = im_processor(images=map_i, return_tensors="pt")['pixel_values']
            im_i = torch.from_numpy(im_group[episode][:]).float().permute(0, 3, 1, 2)
            im_i = F.interpolate(im_i, size=(112,224), mode='bilinear', align_corners=False).squeeze(0)
            im.append(im_i/255.0)
            episode_len = im_i.shape[0]
            a = torch.from_numpy(action_group[episode][:])
            #EOS token
            a = torch.cat((a, torch.ones((1,4))), dim=0)
            actions.append(a)
            a_label = action2label_vocab(a, act_vocab_coords)
            a_labels.append(a_label)    
    
    #check load annotations 
    #print(act_vocab_coords.shape)
    #print(len(act_vocab_im))
    # one im-prompt less, EOS don't have im-prompt
    #print(len(act_vocab_prompt))

    #check dataset load
    #print(len(im))
    #print(len(actions))
    #print(len(a_labels))

    dataset = MyDataset(im=im, prompt=prompt, actions=actions, a_labels = a_labels, map=map)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1-TEST_PART, TEST_PART])
    print('preprocess full time: ',time.time()-preprocess_timer_start)
    print('Starting Training loop...')
    train_loop(train_dataset, test_dataset, act_vocab_coords, act_vocab_im, act_vocab_prompt, act_vocab_map)