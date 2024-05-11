import torch
import h5py
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR
import time
import shutil
import math
from transformers import OpenAIGPTConfig, OpenAIGPTModel
from transformers import OFATokenizer, OFAModel

'''
Behavioral cloning Renas  transformer camera-lidar TRAIN LOOP

State: im-map concatenation (reworked h5), prompt 
states organized as sequences - episodes

Actions in ros: position(x,y) orientation quternions (z, w)
Actions for model are explored (im-prompt description) and set as tokens vocabulary

1. TEXT-Image(camera+map concatenation) encoding using OFA (trainable) 
2. (im_prompt)-(action) causal Transformer GPT 
'''
LR = 10e-6
LR_WARMUP_EPOCHS = 5 
LR_DECAY_EPOCHS = 100

DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/cola/tsa_combined_reworked.h5'
POSES = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/poses/poses_2024-04-25_15-00-52_action_vocab.h5'
TEST_PART = 0.2
BATCH_SIZE = 1
CHECKPOINT_INTERVAL = 10

WEIGHTS_DIR = '/home/renas/pythonprogv2/phd_xiaor_project/weights'
LOAD_WEIGHTS = 'renas7.pt'
SAVE_WEIGHTS = 'renas7.pt'

class StateActionPromptDataset(Dataset):
    def __init__(self, im, action, a_label, prompt):
        self.im = im
        self.action = action
        self.a_label = a_label
        self.prompt = prompt
    def __len__(self):
        return len(self.im)
    def __getitem__(self, idx):
        im = self.im[idx]
        action = self.action[idx]
        a_label = self.a_label[idx]
        prompt = self.prompt[idx]
        return im, action, a_label, prompt
    
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

class Renas(torch.nn.Module):
    def __init__(self, device):
        super(Renas, self).__init__()
        self.device = device
        
        self.tokenizer = OFATokenizer.from_pretrained(ckpt_dir)
        self.ofa_model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
        self.d_model = self.ofa_model.config.d_model
        for param in self.ofa_model.parameters():
            param.requires_grad = False  

        self.pos_enc = PositionalEncoding(d_model=self.d_model)

        self.im_prompt_enc_vector = EncodingVector(d_model=self.d_model)
        self.actions_enc_vector = EncodingVector(d_model=self.d_model)
        
        self.gpt_config = OpenAIGPTConfig(vocab_size=0, n_positions=200, n_embd=self.d_model, n_layer=20, n_head=12)
        self.gpt_model = OpenAIGPTModel(self.gpt_config)



    def forward(self, batch, action_vocab_token):
        im, action, _, prompt = batch
        i1, i2, i3, i4, i5 = im.size()
        im = im.view(i1*i2, i3, i4, i5).to(self.device)

        prompt = [prompt for prompt in prompt for _ in range(i2)]


        prompt = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        #print('here', prompt.shape)
        #print('here2', im.shape)
        im_prompt = self.ofa_model.encoder.forward(input_ids=prompt, patch_images=im)
        im_prompt = im_prompt.last_hidden_state
        #print('here3', im_prompt.shape)
        im_prompt = torch.mean(im_prompt, dim=1).unsqueeze(1)
        im_prompt = im_prompt.view(i1, i2, self.d_model)
        
        actions = action.to(self.device)
        
        im_prompt = self.im_prompt_enc_vector(im_prompt)
        actions = self.actions_enc_vector(actions)

        im_prompt = self.pos_enc(im_prompt)
        actions = self.pos_enc(actions)
        
        # 2 types of data for gpt
        tokens = torch.zeros(i1, i2*2, self.d_model, device=self.device)
        tokens[:, 0::2, :] = im_prompt
        tokens[:, 1::2, :] = actions

        tokens = self.gpt_model(inputs_embeds = tokens).last_hidden_state
        tokens = tokens[:, 0::2, :]
        action_vocab_token = action_vocab_token.to(self.device)
        tokens = self.tokens2similarities(tokens, action_vocab_token) 
        return tokens
    
    def tokens2similarities(self, tokens, action_vocab_token):
        batch_size, seq_length, _ = tokens.shape
        tokens = tokens.reshape(batch_size * seq_length, 1, self.d_model)
        action_vocab_token = action_vocab_token.unsqueeze(0)
        similarity_scores = F.cosine_similarity(tokens, action_vocab_token, dim=2)
        similarity_scores = similarity_scores.reshape(batch_size, seq_length, -1)
        return similarity_scores

def action2token_vocab(action, action_vocab_token, action_vocab_action):
    action = action.unsqueeze(1)
    action_vocab_action = action_vocab_action.unsqueeze(0) 
    similarity_scores = F.cosine_similarity(action, action_vocab_action, dim=2)
    max_values, max_indices = torch.max(similarity_scores, dim=1)
    selected_tokens = [action_vocab_token[idx] for idx in max_indices]
    selected_tokens = torch.stack(selected_tokens, dim=0)
    return selected_tokens

def action2label_vocab(action, action_vocab_action):
    action = action.unsqueeze(1)
    action_vocab_action = action_vocab_action.unsqueeze(0) 
    similarity_scores = F.cosine_similarity(action, action_vocab_action, dim=2)
    max_values, max_indices = torch.max(similarity_scores, dim=1)
    return max_indices

def token2action_vocab(token, action_vocab_token, action_vocab_action):
    token = token.unsqueeze(1)
    action_vocab_token = action_vocab_token.unsqueeze(0) 
    similarity_scores = F.cosine_similarity(token, action_vocab_token, dim=2)
    max_values, max_indices = torch.max(similarity_scores, dim=1)
    selected_actions = [action_vocab_action[idx] for idx in max_indices]
    selected_actions = torch.stack(selected_actions, dim=0)
    return selected_actions

def padding_collate(batch):
    new_batch = []
    for i in range(3): # 4 data types: im, action, a_label, prompt
        #iterate except the last one: prompt
        new_batch.append([item[i] for item in batch])
        new_batch[i] = pad_sequence(new_batch[i], batch_first=True, padding_value= 1.0)
    #add prompt separately without collated torch dataset
    new_batch.append([item[3] for item in batch])
    return new_batch

def train_loop(train_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=padding_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=padding_collate)

    
    model = Renas(device).to(device)
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
            output = model(batch, action_vocab_token)
            if i==0:
                print('correct labels: ', batch[2])
                print('model output: ', output)
            output_flat = output.view(-1, output.shape[-1])
            labels_flat = batch[2].to(device).view(-1)
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
                output = model(batch, action_vocab_token)
                output_flat = output.view(-1, output.shape[-1])
                labels_flat = batch[2].to(device).view(-1)
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
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'temp_'+ SAVE_WEIGHTS))
            shutil.move(os.path.join(WEIGHTS_DIR, 'temp_'+ SAVE_WEIGHTS), os.path.join(WEIGHTS_DIR, SAVE_WEIGHTS))

            if test_average_loss.item()<min_loss:
                torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'temp_'+ 'early_'+ SAVE_WEIGHTS))
                shutil.move(os.path.join(WEIGHTS_DIR, 'temp_'+'early_'+ SAVE_WEIGHTS), os.path.join(WEIGHTS_DIR, 'early_'+ SAVE_WEIGHTS))
                min_loss = test_average_loss.item()
                print('Early stopping with loss', min_loss, 'at the epoch', epoch)
            print('weights saved')




















if __name__ == '__main__':
    ckpt_dir = '/home/renas/pythonprogv2/phd_xiaor_project/OFA-base'
    im = []
    action = []
    a_label = []
    prompt = []
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device('cuda:0')
            device_i = torch.device(f'cuda:{i}')
            print(f'Cuda Device {i}: ', device_i, torch.cuda.get_device_name(i))
    else:
        print('No CUDA devices available')
        device = torch.device('cpu')
    print('Current device: ',device)

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

    with h5py.File(DATASET, 'r') as hdf:
        num_episodes = len(hdf['states'])
        for i in range(num_episodes):
            episode_i = 'data_'+str(i)
            im_i = torch.from_numpy(hdf['states'][episode_i][:]).float()
            im.append(im_i)

            a = torch.from_numpy(hdf['actions'][episode_i][:])
            a = torch.cat((a, torch.ones((1,4))), dim=0)
            a_label_i = action2label_vocab(a, action_vocab_action)
            a_label.append(a_label_i)
            a = action2token_vocab(a, action_vocab_token, action_vocab_action)
            action.append(a)
    
    prompt_filename = DATASET[:-12]+'_tasks.txt'
    with open(prompt_filename, 'r') as file:
        for p in file:
            prompt.append(p.strip())

    dataset =  StateActionPromptDataset(im, action, a_label, prompt)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1-TEST_PART, TEST_PART])
    train_loop(train_dataset, test_dataset)