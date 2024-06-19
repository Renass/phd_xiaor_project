from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
import h5py
from torch.utils.data import Dataset, DataLoader

'''
Try instructblip to my camera-lidar dataset

State: im-map concatenation (reworked h5), prompt 
states organized as sequences - episodes

Actions in ros: position(x,y) orientation quternions (z, w)
Actions for model are explored (im-prompt description) and set as tokens vocabulary
'''

DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/real/2A724_may/tsa_combined_reworked.h5'
EPISODE_NUM = 40
PROMPT = 'Find a green cone'

class SimpleDataset(Dataset):
    def __init__(self, im):
        self.im = im
    def __len__(self):
        return len(self.im)
    def __getitem__(self, idx):
        im = self.im[idx]
        return im

if __name__ == '__main__':
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", torch_dtype=torch.float16)
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    print(dir(processor.image_processor))
    print(processor.image_processor.do_rescale)
    processor.image_processor.do_rescale = False
    processor.image_processor.do_resize = True
    processor.image_processor.do_normalize = False
    print(processor.image_processor.do_rescale)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    #print(model.language_model)
    #print(model.get_output_embeddings)

    im = []
    with h5py.File(DATASET, 'r') as hdf:
        num_episodes = len(hdf['states'])
        print('num_episodes:', num_episodes)
        for i in range(num_episodes):
            episode_i = 'data_'+str(i)
            im_i = torch.from_numpy(hdf['states'][episode_i][:]).float()
            im.append(im_i)
    prompt = PROMPT
    image = im[EPISODE_NUM]
    image = image[-1]
    print('here')
    print(image.shape)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    batch_size = inputs['input_ids'].size(0)

    #print('here', model.config)
    #decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids
    # Initialize decoder_input_ids with the BOS token
    if 'decoder_input_ids' not in inputs:
        inputs['decoder_input_ids'] = torch.LongTensor([model.config.text_config.bos_token_id]).repeat(batch_size, 1).to(inputs['input_ids'].device)


    outputs = model.forward(**inputs, return_dict=True)
    print(outputs.language_model_outputs.encoder_last_hidden_state.shape)
    outputs = model.generate(
            **inputs,
            do_sample=True,
            num_beams=5,
            max_length=256,
            min_length=5,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(generated_text)