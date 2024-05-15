from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", load_in_4bit=True)
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")

device = "cuda" if torch.cuda.is_available() else "cpu"
#model.to(device)
model.eval()
#print(model.language_model)
#print(model.get_output_embeddings)

im = "/home/renas/pythonprogv2/phd_xiaor_project/camera-lidar_experiment/Figure_3.png"
image = Image.open(im).convert("RGB")
prompt = "Task: Go to the fridge"


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
        do_sample=False,
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