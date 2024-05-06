import h5py
import torch
import pandas as pd
from torch.nn.functional import cosine_similarity

def load_embeddings(file_path):
    tokens = []
    actions = []
    with h5py.File(file_path, 'r') as hdf:
        num_datasets = len(hdf['tokens'])
        for i in range(num_datasets):
            token_data = torch.tensor(hdf['tokens'][f'data_{i}'][:])
            action_data = torch.tensor(hdf['actions'][f'data_{i}'][:])
            tokens.append(token_data)
            actions.append(action_data)

    # Stack all tokens and actions
    tokens = torch.stack(tokens, dim=0)
    actions = torch.stack(actions, dim=0)
    
    return tokens, actions

# Compute cosine similarity matrix
def compute_similarity_matrix(embeddings):
    similarity_matrix = cosine_similarity(embeddings, embeddings)
    return similarity_matrix

# Convert to DataFrame for visualization
def similarity_matrix_to_dataframe(matrix):
    df = pd.DataFrame(matrix.numpy())
    return df

# Path to the uploaded HDF5 file
file_path = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/real/poses/poses_2024-05-04_18-10-20_action_vocab.h5'

# Perform operations
embeddings = load_embeddings(file_path)
similarity_matrix = compute_similarity_matrix(embeddings)
similarity_df = similarity_matrix_to_dataframe(similarity_matrix)
similarity_df
