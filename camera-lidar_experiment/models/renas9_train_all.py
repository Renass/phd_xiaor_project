import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, InstructBlipConfig
import h5py
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import bfloat16
import torch.nn.functional as F
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import OpenAIGPTConfig, OpenAIGPTModel
from transformers import BertTokenizer, BertModel, BertConfig
import math
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
from torch.utils.tensorboard import SummaryWriter
import os
import time
import inspect
import shutil

'''
TRAIN LOOP for Renas MODEL 9 with multimodal encoder Included in training
File work:
    input:
        tsa_combined.h5 (demonstrations dataset)
        tsa_combined_tasks.txt (demonstrations dataset task prompts)
        action_annotation.h5 - image descriptions of action options
        action_annotation_tasks.txt - prompt annotations of action options 
   
MODEL 9:
    Behavioral cloning Renas  transformer camera-lidar
    1. TEXT-Image camera or (camera+map concatenation) ENCODER using InstructBLIP 
    2. TEXT-Image camera or (camera+map concatenation) DECODER using InstructBLIP for text generation
    3. Cross-attention middle tokens to cls driving token MID TRANSFORMER
    4. (im_prompt)-(action) history-aware causal driving Transformer GPT
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

DATASET = '/data/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/real/2A724_may/tsa_combined.h5'
DEVICE = 'cuda:0'
ACTION_ANNOTATION = '/data/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/real/poses/poses_2024-05-04_18-10-20.h5'

LR = 10e-7
LR_WARMUP_EPOCHS = 5 
LR_DECAY_EPOCHS = 100

TEST_PART = 0.2
BATCH_SIZE = 1
CHECKPOINT_INTERVAL = 25

WEIGHTS_DIR = '/data/renas/pythonprogv2/phd_xiaor_project/weights'
LOAD_WEIGHTS = 'renas9.pt'
SAVE_WEIGHTS = 'renas9_all.pt'

class MyDataset(Dataset):
    def __init__(self, im, prompt, actions, a_labels):
        self.im = im
        self.prompt = prompt
        self.actions = actions
        self.a_labels = a_labels
    def __len__(self):
        return len(self.im)
    def __getitem__(self, idx):
        im = self.im[idx]
        prompt = self.prompt[idx]
        action = self.actions[idx]
        a_label = self.a_labels[idx]
        return im, prompt, action, a_label

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
    
class EncodingVector(torch.nn.Module):
    def __init__(self, d_model):
        super(EncodingVector, self).__init__()
        self.modality_vector = torch.nn.Parameter(torch.randn(d_model))
    def forward(self, x):
        return x + self.modality_vector.unsqueeze(0).unsqueeze(0)

class Renas9forTrain(torch.nn.Module):
    def __init__(self, device):
        super(Renas9forTrain, self).__init__()
        self.device = device
        #self.d_model = 2048

        self.blip_config = InstructBlipConfig.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        self.d_model = self.blip_config.text_config.d_model

        self.blip_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        self.blip_processor.image_processor.do_rescale = True
        self.blip_processor.image_processor.do_resize = True
        self.blip_processor.image_processor.do_normalize = False

        self.blip_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", torch_dtype=torch.bfloat16)
        for param in self.blip_model.parameters():
            param.requires_grad = True 


        self.mid_t_config = BertConfig( 
            hidden_size=self.d_model, 
            intermediate_size=self.d_model*4,
            num_hidden_layers= 6,
            num_attention_heads= 32
            )
        self.mid_t_model = BertModel(config=self.mid_t_config).to(torch.bfloat16)


        self.pos_enc = PositionalEncoding(d_model=self.d_model)
        self.im_prompt_enc_vector = EncodingVector(d_model=self.d_model).to(torch.bfloat16)
        self.actions_enc_vector = EncodingVector(d_model=self.d_model).to(torch.bfloat16)
        
        self.gpt_config = OpenAIGPTConfig(vocab_size=0, n_positions=200, n_embd=self.d_model, n_layer=6, n_head=32)
        self.gpt_model = OpenAIGPTModel(self.gpt_config).to(torch.bfloat16)

        #Weights for final cross-attention multiple choice
        self.q_weights = torch.nn.Linear(self.d_model, self.d_model, dtype=torch.bfloat16)
        self.k_weights = torch.nn.Linear(self.d_model, self.d_model, dtype=torch.bfloat16)



    def forward(self, batch, act_vocab_coords, act_vocab_im, act_vocab_prompt):
        
        #BLIP encoder
        #Work with action annotation
        act_vocab_token = []
        for i, im_i in enumerate(act_vocab_im):
            inputs = self.blip_processor(images=im_i, text= act_vocab_prompt[i], return_tensors="pt")
            inputs = {key: val.to(device) for key, val in inputs.items()}
            if 'decoder_input_ids' not in inputs:
                inputs['decoder_input_ids'] = torch.LongTensor([self.blip_config.text_config.bos_token_id]).repeat(1, 1).to(inputs['input_ids'].device)
            outputs = self.blip_model.forward(**inputs, return_dict=True)
            #print(outputs.language_model_outputs.encoder_last_hidden_state.dtype)
            #print('Action annotation token shape:', outputs.language_model_outputs.encoder_last_hidden_state.shape)
            act_vocab_token.append(outputs.language_model_outputs.encoder_last_hidden_state)     
        #For EOS token
        act_vocab_token.append(torch.ones_like(act_vocab_token[0]))
        #BLIP encoder
        #work with dataset
        im, prompt, action, _ = batch
        state = []
        for i, im_i in enumerate(im):
            episode_len = im_i.shape[0]
            inputs = self.blip_processor(images=[im_i[j] for j in range(episode_len)], text=[prompt[i]]*episode_len, return_tensors="pt")
            inputs = {key: val.to(device) for key, val in inputs.items()}

            if 'decoder_input_ids' not in inputs:
                inputs['decoder_input_ids'] = torch.LongTensor([self.blip_config.text_config.bos_token_id]).repeat(episode_len, 1).to(inputs['input_ids'].device)
            outputs = self.blip_model.forward(**inputs, return_dict=True)
            #print('states shape:', outputs.language_model_outputs.encoder_last_hidden_state.shape)
            outputs.language_model_outputs.encoder_last_hidden_state
            state.append(outputs.language_model_outputs.encoder_last_hidden_state)
             
        #mid-transformer 
        # action annotation
        an = []
        for i in act_vocab_token:
            i = i.to(self.device)
            attention_mask = torch.ones(i.size()[:-1], dtype=torch.long).to(self.device)
            an.append(self.mid_t_model.forward(inputs_embeds = i, attention_mask= attention_mask).pooler_output)            
        an = torch.cat(an, dim=0)
        #mid-transformer
        #action
        action = action2token_vocab(action, act_vocab_coords, act_vocab_token)
        aa= []
        for i in action:
            a = []
            for j in i:
                j = j.to(self.device)
                attention_mask = torch.ones(j.size()[:-1], dtype=torch.long).to(self.device)
                a.append(self.mid_t_model.forward(inputs_embeds = j, attention_mask= attention_mask).pooler_output)
            aa.append(torch.cat(a, dim=0))
        action= torch.stack(aa, dim=0)
        #mid-transformer
        #state
        ss = [] 
        for i in state:
            s = []
            for j in i:
                j = j.unsqueeze(0)
                attention_mask = torch.ones(j.size()[:-1], dtype=torch.long).to(self.device)
                s.append(self.mid_t_model.forward(inputs_embeds = j, attention_mask= attention_mask).pooler_output)
            ss.append(torch.cat(s, dim=0))
        state = torch.stack(ss, dim=0)


        #state-action-gpt part
        state = self.im_prompt_enc_vector(state)
        action = self.actions_enc_vector(action)
        #state = self.pos_enc(state)
        #action = self.pos_enc(action)
        
        batch_size, seq_len, _ = state.shape
        # 2 types of data for gpt
        tokens = torch.zeros(batch_size, seq_len*2, self.d_model, device=self.device, dtype=torch.bfloat16)
        tokens[:, 0::2, :] = state
        tokens[:, 1::2, :] = action

        tokens = self.gpt_model(inputs_embeds = tokens).last_hidden_state
        tokens = tokens[:, 0::2, :]
        tokens = self.q_weights(tokens)
        an = self.k_weights(an).unsqueeze(0)
        attention_scores = torch.matmul(tokens, an.transpose(1, 2))
        return attention_scores
    
def action2token_vocab(action, act_vocab_coords, act_vocab_tokens):
    '''
    Convert action on coordinates (quaternions) shaped:
    [seq_length, 4] or [batch_size, seq_length, 4]
    to action in encoder representation (batch_size, seq_length, d_model) 
    '''
    #correct shape: [1, vocab_size, 4]
    act_vocab_coords = act_vocab_coords.unsqueeze(0)

    if action.dim() == 2:
        action = action.unsqueeze(0)
    batch_size = action.shape[0] 
    
    # Compute cosine similarity (batch_size, seq_length, vocab_size)
    similarity_scores = F.cosine_similarity(action.unsqueeze(2), act_vocab_coords.unsqueeze(0), dim=-1)
    
    # Find the max similarity scores and their indices
    max_values, max_indices = torch.max(similarity_scores, dim=-1)
    #print('here', max_indices)
    if torch.min(max_values).item() < 0.999:
        print('Warning: action coordinates not match vocabulary!!!')
    a = []
    aa= []
    for i in range(batch_size):
        for j in max_indices[i]:
            a.append(act_vocab_tokens[j])
        aa.append(a)
    return aa

def action2label_vocab(action, action_vocab_action):
    action = action.unsqueeze(1)
    action_vocab_action = action_vocab_action.unsqueeze(0) 
    similarity_scores = F.cosine_similarity(action, action_vocab_action, dim=2)
    max_values, max_indices = torch.max(similarity_scores, dim=1)
    if torch.min(max_values).item() < 0.999:
        print('Warning: action coordinates not match vocabulary!!!')
    return max_indices

def padding_collate(batch):
    #num_data_types = len(batch[0])
    new_batch = []
    for i in range(4): # 4 data types: states, prompt, action, a_label (skip collate for prompt)
        new_batch.append([item[i] for item in batch])
        if i != 1:
            new_batch[i] = pad_sequence(new_batch[i], batch_first=True, padding_value= 1.0)
    return new_batch

def train_loop(train_dataset, test_dataset, act_vocab_coords, act_vocab_tokens, act_vocab_prompt):
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=padding_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=padding_collate)    
    model = Renas9forTrain(DEVICE).to(DEVICE)
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
        for i, batch in enumerate(train_dataloader):
            output = model(batch, act_vocab_coords, act_vocab_im, act_vocab_prompt)
            # prinn 1st batch 1st episode labels and predictions
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
                output = model(batch, act_vocab_coords, act_vocab_im, act_vocab_prompt)
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
    actions = []
    prompt = []
    act_vocab_prompt = []
    act_vocab_im = []
    act_vocab_coords = []
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
    with h5py.File(ACTION_ANNOTATION, 'r') as annot_hdf:
        im_group = annot_hdf['states']
        action_group = annot_hdf['actions']
        num_annots = len(im_group)
        print('ACTION ANNOTATION contains options: ', num_annots)
        for i in range(num_annots+1):
            if i<num_annots:
                #For annons except EOS token
                annot = 'data_'+str(i)
                im_i = torch.from_numpy(im_group[annot][0]).float()
                act_vocab_im.append(im_i)   
                act_vocab_coords.append(torch.from_numpy(action_group[annot][0]))
            else:
                #For EOS token
                #act_vocab_im.append(torch.ones_like(act_vocab_im[0]))
                act_vocab_coords.append(torch.ones_like(act_vocab_coords[0]))
        act_vocab_coords = torch.stack(act_vocab_coords, dim=0)
                        
    #load demonstrations dataset
    with h5py.File(DATASET, 'r') as hdf:
        im_group = hdf['states']
        action_group = hdf['actions']
        num_episodes = len(im_group)
        print('Dataset contains episodes: ', num_episodes)
        for i in range(num_episodes):
            episode = 'data_'+str(i)
            im_i = torch.from_numpy(im_group[episode][:]).float()
            im.append(im_i)
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

    dataset = MyDataset(im=im, prompt=prompt, actions=actions, a_labels = a_labels)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1-TEST_PART, TEST_PART])
    print('preprocess full time: ',time.time()-preprocess_timer_start)
    print('Starting Training loop...')
    train_loop(train_dataset, test_dataset, act_vocab_coords, act_vocab_im, act_vocab_prompt)