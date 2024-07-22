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
TRAIN LOOP for Renas MODEL 9
File work:
    input:
        _model9_prep.h5 (demonstartion episodes, where states-actions made as ENCODER context - sequences of tokens):
            'states' : blip2encoder representations of states
            'actions': actions in 4 coordinates(reduced quaternion)
            'act_vocab_tokens' : blip2encoder representations of action vocabulary
            'act_vocab_coords' : 4 coordinates (reduced quaternion) action vocabulary 
   
MODEL 9:
    Behavioral cloning Renas  transformer camera-lidar
    1. TEXT-Image camera or (camera+map concatenation) ENCODER using InstructBLIP (frozen) 
    2. TEXT-Image camera or (camera+map concatenation) DECODER using InstructBLIP (frozen) for text generation
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

LR = 10e-7
LR_WARMUP_EPOCHS = 5 
LR_DECAY_EPOCHS = 100

DEVICE = 'cuda:0'
TEST_PART = 0.2
# DATASET is preprocessed with renas9prep.py file 
DATASET = '/data/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/real/2A724_may/tsa_combined_model9_prep.h5'
BATCH_SIZE = 1
CHECKPOINT_INTERVAL = 25

WEIGHTS_DIR = '/data/renas/pythonprogv2/phd_xiaor_project/weights'
LOAD_WEIGHTS = 'early_renas9.pt'
SAVE_WEIGHTS = 'renas9.pt'

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
        self.d_model = 2048

        self.mid_t_config = BertConfig( 
            hidden_size=self.d_model, 
            intermediate_size=self.d_model*4,
            num_hidden_layers= 20,
            num_attention_heads= 32
            )
        self.mid_t_model = BertModel(config=self.mid_t_config)


        self.pos_enc = PositionalEncoding(d_model=self.d_model)
        self.im_prompt_enc_vector = EncodingVector(d_model=self.d_model)
        self.actions_enc_vector = EncodingVector(d_model=self.d_model)
        
        self.gpt_config = OpenAIGPTConfig(vocab_size=0, n_positions=200, n_embd=self.d_model, n_layer=20, n_head=32)
        self.gpt_model = OpenAIGPTModel(self.gpt_config)

        #Weights for final cross-attention multiple choice
        self.q_weights = torch.nn.Linear(self.d_model, self.d_model)
        self.k_weights = torch.nn.Linear(self.d_model, self.d_model)



    def forward(self, batch, act_vocab_coords, act_vocab_token):
        state, action, _, = batch
        state = state.to(self.device)
        action = action.to(self.device)
        act_vocab_coords = act_vocab_coords.to(self.device)
        
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
        tokens = torch.zeros(batch_size, seq_len*2, self.d_model, device=self.device)
        tokens[:, 0::2, :] = state
        tokens[:, 1::2, :] = action

        tokens = self.gpt_model(inputs_embeds = tokens).last_hidden_state
        tokens = tokens[:, 0::2, :]
        tokens = self.q_weights(tokens)
        an = self.k_weights(an).unsqueeze(0)
        attention_scores = torch.matmul(tokens, an.transpose(1, 2))
        return attention_scores

class StateActionDataset(Dataset):
    def __init__(self, state, action, a_label):
        self.state = state
        self.action = action
        self.a_label = a_label
    def __len__(self):
        return len(self.state)
    def __getitem__(self, idx):
        state = self.state[idx]
        action = self.action[idx]
        a_label = self.a_label[idx]
        return state, action, a_label

def action2label_vocab(action, action_vocab_action):
    action = action.unsqueeze(1)
    action_vocab_action = action_vocab_action.unsqueeze(0) 
    similarity_scores = F.cosine_similarity(action, action_vocab_action, dim=2)
    max_values, max_indices = torch.max(similarity_scores, dim=1)
    return max_indices

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

def padding_collate(batch):
    new_batch = []
    for i in range(3): # 3 data types: states, action, a_label
        new_batch.append([item[i] for item in batch])
        new_batch[i] = pad_sequence(new_batch[i], batch_first=True, padding_value= 1.0)
    return new_batch

def print_shape(x):
    frame = inspect.currentframe().f_back
    variable_names = {id(value): name for name, value in frame.f_locals.items()}
    var_name = variable_names.get(id(x), 'variable')
    print(var_name, 'shape:', x.shape)

def main():
    states = []
    actions = []
    a_labels = []
    act_vocab_tokens = []
    act_vocab_coords = []
    global device

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device(DEVICE)
            device_i = torch.device(f'cuda:{i}')
            print(f'Cuda Device {i}: ', device_i, torch.cuda.get_device_name(i))
    else:
        print('No CUDA devices available')
        device = torch.device('cpu')
    print('Current device: ',device)

    #load data
    with h5py.File(DATASET, 'r') as hdf:
        num_episodes = len(hdf['states'])
        num_annots = len(hdf['act_vocab_tokens'])
        print('Dataset contains episodes: ', num_episodes)
        print('Action vocabulary contains options (with EOS): ', num_annots)
        for i in range(num_annots):
            annot_i = 'data_'+str(i)
            annot = torch.from_numpy(hdf['act_vocab_tokens'][annot_i][:]).to(dtype=torch.bfloat16)
            act_vocab_tokens.append(annot)
            annot = torch.from_numpy(hdf['act_vocab_coords'][annot_i][:]).float()
            act_vocab_coords.append(annot)
        act_vocab_coords = torch.stack(act_vocab_coords, dim=0)
        for i in range(num_episodes):
            episode_i = 'data_'+str(i)
            state = torch.from_numpy(hdf['states'][episode_i][:]).to(dtype=torch.bfloat16)
            states.append(state)
            action = torch.from_numpy(hdf['actions'][episode_i][:]).float()
            #print(action)
            actions.append(action)
            a_label = action2label_vocab(action, act_vocab_coords)
            #print(a_label)
            a_labels.append(a_label)

    
    dataset =  StateActionDataset(states, actions, a_labels)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1-TEST_PART, TEST_PART])
    print('Starting Training loop...')
    train_loop(train_dataset, test_dataset, act_vocab_coords, act_vocab_tokens)

def train_loop(train_dataset, test_dataset, act_vocab_coords, act_vocab_tokens):
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=padding_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=padding_collate)
    
    model = Renas9forTrain(device).to(device)
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
            output = model(batch, act_vocab_coords, act_vocab_tokens)
            # prinn 1st batch 1st episode labels and predictions
            if i==0:
                print('correct labels: ', batch[2][0])
                print('model output: ')
                softmax_output = F.softmax(output[0], dim=-1)
                formatted_output = [[f"{value.item():.2f}" for value in row] for row in softmax_output]
                for row in formatted_output:
                    print(row)
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
                output = model(batch, act_vocab_coords, act_vocab_tokens)
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
    main()