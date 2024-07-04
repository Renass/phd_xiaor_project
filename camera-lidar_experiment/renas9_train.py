import torch
import h5py

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

DEVICE = 'cuda:0'
# DATASET is preprocessed with renas9prep.py file 
DATASET = '/data/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/real/2A724_may/tsa_combined_model9_prep.h5'

def main():
    states = []
    actions = []
    a_label = []

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
        print('Action vocabulary contains options: ', num_annots)
        for i in range(num_episodes):
            episode_i = 'data_'+str(i)
            state = torch.from_numpy(hdf['states'][episode_i][:]).to(dtype=torch.bfloat16)
            states.append(state)
            action = torch.from_numpy(hdf['actions'][episode_i][:]).float()
            actions.append(action)

            print('\n')
            print(state.shape)
            print(action.shape)

if __name__ == '__main__':
    main()