from transformers import AutoProcessor, AutoModel
import torch

processor = AutoProcessor.from_pretrained("allenai/ufo-vqa")
model = AutoModel.from_pretrained("allenai/ufo-vqa")

image = torch.randn(1, 3, 224, 224) 
prompt = "What is in the image?"

inputs = processor(images=image, text=prompt, return_tensors="pt")
outputs = model(**inputs)
cls_token = outputs.pooler_output  # (batch_size, hidden_dim)