import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, InstructBlipConfig
import h5py
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import bfloat16
import torch.nn.functional as F

'''
PREPROCESING BLIP ENCODER to multiple context moment-wise state-action tokens
File work:
    input:
        tsa_combined.h5 (demonstrations dataset)
        tsa_combined_tasks.txt (demonstrations dataset task prompts)
        action_annotation.h5 - image descriptions of action options
        action_annotation_tasks.txt - prompt annotations of action options
    output:
        _model9_prep.h5 (demonstartion episodes, where states-actions made as ENCODER context - sequences of tokens):

   
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

DATASET = '/data/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/real/2A724_may/tsa_combined.h5'
DEVICE = 'cuda:0'
ACTION_ANNOTATION = '/data/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/real/poses/poses_2024-05-04_18-10-20.h5'

def look_im(im_i, generated_text):
    im_i_np = im_i.numpy()
    plt.imshow(im_i_np.astype(np.uint8))
    plt.title(generated_text)
    plt.axis('off')
    plt.show()



class Renas9(torch.nn.Module):
    def __init__(self, device):
        super(Renas9, self).__init__()
        self.device = device
        self.blip_config = InstructBlipConfig.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        self.d_model = self.blip_config.text_config.d_model
        
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        self.processor.image_processor.do_rescale = True
        self.processor.image_processor.do_resize = True
        self.processor.image_processor.do_normalize = False

        self.blip_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", torch_dtype=torch.bfloat16)
        for param in self.blip_model.parameters():
            param.requires_grad = False 

        


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

    new_dataset_path = DATASET[:-3]+'_model9_prep.h5'
    model = Renas9(DEVICE).to(DEVICE)

    im = []
    action = []
    prompt = []
    annot_prompt = []
    action_vocab_token = []
    action_vocab_coordinate = []

    prompt_filename = DATASET[:-3]+'_tasks.txt'
    with open(prompt_filename, 'r') as file:
        for p in file:
            prompt.append(p.strip())

    annot_prompt_filename = ACTION_ANNOTATION[:-3]+'_tasks.txt'
    with open(annot_prompt_filename, 'r') as file:
        for p in file:
            annot_prompt.append(p.strip())
    print(annot_prompt)

    with h5py.File(new_dataset_path, 'w') as new_hdf:
        new_hdf_im_group = new_hdf.create_group('states')
        new_hdf_action_group = new_hdf.create_group('actions')
        new_hdf_act_vocab_tokens_group = new_hdf.create_group('act_vocab_tokens')
        new_hdf_act_vocab_coordinates_group = new_hdf.create_group('act_vocab_coords')
        
        #From action annotations create action-token vocabulary
        with h5py.File(ACTION_ANNOTATION, 'r') as annot_hdf:
            im_group = annot_hdf['states']
            action_group = annot_hdf['actions']
            num_annots = len(im_group)
            print('ACTION ANNOTATION contains options: ', num_annots)
            for i in range(num_annots):
                annot = 'data_'+str(i)
                im_i = torch.from_numpy(im_group[annot][0]).float()
                inputs = model.processor(images=im_i, text= prompt[i], return_tensors="pt")
                inputs = {key: val.to(device) for key, val in inputs.items()}
                if 'decoder_input_ids' not in inputs:
                    inputs['decoder_input_ids'] = torch.LongTensor([model.blip_config.text_config.bos_token_id]).repeat(1, 1).to(inputs['input_ids'].device)
                outputs = model.blip_model.forward(**inputs, return_dict=True)
                #print(outputs.language_model_outputs.encoder_last_hidden_state.dtype)
                action_vocab_token.append(outputs.language_model_outputs.encoder_last_hidden_state)   
                action_vocab_coordinate.append(torch.from_numpy(action_group[annot][0]))
                new_hdf_act_vocab_tokens_group.create_dataset('data_'+str(i), data=action_vocab_token[i].cpu().to(dtype=torch.float32), dtype = np.float32, compression = 'gzip')
                new_hdf_act_vocab_coordinates_group.create_dataset('data_'+str(i), data=action_vocab_coordinate[i], dtype = np.float32, compression = 'gzip')
        
        action_vocab_token = torch.stack(action_vocab_token, dim=0)
        #additional end_token of ones
        action_vocab_token = torch.cat((action_vocab_token, torch.ones_like(action_vocab_token[0].unsqueeze(0))))
        
        action_vocab_coordinate = torch.stack(action_vocab_coordinate, dim=0)
        #additional end_token of ones
        action_vocab_coordinate = torch.cat((action_vocab_coordinate, torch.ones((1, 4))))  
        
        with h5py.File(DATASET, 'r') as hdf:
            im_group = hdf['states']
            action_group = hdf['actions']
            num_episodes = len(im_group)
            print('Dataset contains episodes: ', num_episodes)
            for i in range(num_episodes):
                episode = 'data_'+str(i)
                im_i = torch.from_numpy(im_group[episode][:]).float()
                episode_len = im_i.shape[0]
                inputs = model.processor(images=[im_i[j] for j in range(episode_len)], text=[prompt[i]]*episode_len, return_tensors="pt")
                inputs = {key: val.to(device) for key, val in inputs.items()}

                if 'decoder_input_ids' not in inputs:
                    inputs['decoder_input_ids'] = torch.LongTensor([model.blip_config.text_config.bos_token_id]).repeat(episode_len, 1).to(inputs['input_ids'].device)
                outputs = model.blip_model.forward(**inputs, return_dict=True)
                #print('Last hidden state: ', outputs.language_model_outputs.encoder_last_hidden_state.shape)
                new_hdf_im_group.create_dataset(episode, data=outputs.language_model_outputs.encoder_last_hidden_state.cpu().to(dtype=torch.float32), dtype = np.float32, compression = 'gzip')
                    #outputs = model.blip_model.generate(
                    #        **inputs,
                    #        do_sample=True,
                    #        num_beams=5,
                    #        max_length=512,
                    #        min_length=10,
                    #        top_p=0.9,
                    #        repetition_penalty=2.5,
                    #        length_penalty=0.5,
                    #        temperature=1,
                    #)
                    #generated_text = model.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                    #print('\n'+generated_text)
                    #look_im(im_i, generated_text)
                
                a = torch.from_numpy(action_group[episode][:])
                a = torch.cat((a, torch.ones((1,4))), dim=0)
                
                print(a.shape)
                #new_hdf_action_group.create_dataset(episode, data=a, dtype = np.float32, compression = 'gzip')

    
    print('preprocess full time: ',time.time()-preprocess_timer_start)
    