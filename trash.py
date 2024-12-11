from transformers import InstructBlipProcessor

# Initialize the InstructBlip processor
processor = InstructBlipProcessor.from_pretrained('Salesforce/instructblip-flan-t5-xl')

# Your input tensor of token IDs
tokens = [
    [837, 3316, 10, 3, 2, 8221, 3155, 1102, 5008, 11416, 11, 7782, 71, 4256, 13582, 9156, 10, 1],
    [1263, 12, 8, 11416, 6, 258, 281, 12, 8, 269, 6358, 1067, 8, 7690, 5, 156, 132, 19, 3, 9, 5571, 14075, 6, 919, 646, 596, 13, 8, 6358, 1],
]

# Decode the token IDs back to text using the processor's tokenizer
for token_ids in tokens:
    decoded_text = processor.tokenizer.decode(token_ids, skip_special_tokens=False)
    print(decoded_text)
