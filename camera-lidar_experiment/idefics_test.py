import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
'''
Behavioral cloning Renas  transformer camera-lidar TESTING

???State: im, map (single per dataset), NO costmap, pose, mapinfo (single per dataset), prompt

Actions in ros: position(x,y) orientation quternions (z, w)

1. ???
2. >Text-Image token + lidar map, NO costmap, pose self-attention transformer 
3. (State)-(action) causal Transformer GPT 

'''

DATASET = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/nav/real/tsa-trajs_2024-04-05_21-03-28.h5'
PROMPTS = ['<mage>What is on the image?',]




if __name__ == '__main__':

    from transformers import CONFIG_MAPPING
    print(list(CONFIG_MAPPING.keys()))

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Cuda Device: ',device, torch.cuda.get_device_name(device))
    else:
        device = torch.device('cpu')
    print('Current device: ',device)

    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-base")
    model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b-base",
    ).to(device)

    image = load_image('https://cdn.pixabay.com/photo/2016/05/05/02/37/sunset-1373171_1280.jpg')

    inputs = processor(text=PROMPTS, images=image, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    print(generated_texts)