import torch
import h5py
import numpy as np

# Create a list of PyTorch tensors with bfloat16
action_vocab_coordinate = [torch.randn(5, 5, dtype=torch.bfloat16).cuda() for _ in range(5)]

# File name for the HDF5 file
file_name = 'tensors_experiment_bfloat16.h5'

# Function to save the list of tensors to HDF5
def save_tensors_to_hdf5(tensor_list, file_name):
    with h5py.File(file_name, 'w') as f:
        # Create a group
        new_hdf_act_vocab_coordinates_group = f.create_group('action_vocab_coordinates')
        
        # Iterate over the list and save each tensor as a dataset
        for i, tensor in enumerate(tensor_list):
            # Ensure the tensor is on the CPU and convert to float32
            cpu_tensor = tensor.cpu().to(dtype=torch.float32).numpy()
            # Save to HDF5
            new_hdf_act_vocab_coordinates_group.create_dataset(f'data_{i}', data=cpu_tensor, compression='gzip')

# Function to load the list of tensors from HDF5
def load_tensors_from_hdf5(file_name):
    with h5py.File(file_name, 'r') as f:
        new_hdf_act_vocab_coordinates_group = f['action_vocab_coordinates']
        loaded_tensors = []
        for i in range(len(new_hdf_act_vocab_coordinates_group)):
            data = new_hdf_act_vocab_coordinates_group[f'data_{i}'][()]
            tensor = torch.tensor(data, dtype=torch.float32).to(dtype=torch.bfloat16)
            loaded_tensors.append(tensor)
    return loaded_tensors

# Save the tensors to HDF5
save_tensors_to_hdf5(action_vocab_coordinate, file_name)

# Load the tensors back from HDF5
loaded_action_vocab_coordinate = load_tensors_from_hdf5(file_name)

# Verify that the loaded tensors match the original tensors
for original, loaded in zip(action_vocab_coordinate, loaded_action_vocab_coordinate):
    # Ensure both tensors are on the same device (CPU for comparison)
    original_cpu = original.cpu()
    loaded_cpu = loaded.cpu()
    assert torch.equal(original_cpu, loaded_cpu), "Mismatch between original and loaded tensor"

print("All tensors saved and loaded correctly.")


