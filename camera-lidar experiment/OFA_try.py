from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
import torch

ckpt_dir = '/home/renas/pythonprogv2/phd_xiaor_project/OFA-base'



mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 384
patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        #transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
    ])


tokenizer = OFATokenizer.from_pretrained(ckpt_dir)

txt = "Describe the image"
inputs = tokenizer([txt], return_tensors="pt").input_ids
img = Image.open('/home/renas/Figure_1.png')
patch_img = patch_resize_transform(img).unsqueeze(0)

model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
for param in model.parameters():
    param.requires_grad = False 
#gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3) 
#model.forward(inputs, patch_images=patch_img)
#print(tokenizer.batch_decode(gen, skip_special_tokens=True))

outputs = model.encoder.forward(input_ids=inputs, patch_images=patch_img)


encoder_last_hidden_state = outputs.last_hidden_state
cls = torch.mean(encoder_last_hidden_state, dim=1)
print(encoder_last_hidden_state.shape)