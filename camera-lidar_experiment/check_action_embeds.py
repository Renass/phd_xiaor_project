import h5py
import torch
import pandas as pd
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

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

def compute_similarity_matrix(embeddings):
    # Expand embeddings along dim=0 and dim=1 to form a grid for comparison
    expanded_embeddings_1 = embeddings.unsqueeze(1)  # Shape: (num_embeddings, 1, embedding_dim)
    expanded_embeddings_2 = embeddings.unsqueeze(0)  # Shape: (1, num_embeddings, embedding_dim)

    # Calculate cosine similarity along the last dimension (embedding_dim)
    similarity_matrix = F.cosine_similarity(expanded_embeddings_1, expanded_embeddings_2, dim=-1)
    return similarity_matrix

def compute_euclidean_distance_matrix(embeddings):
    expanded_embeddings_1 = embeddings.unsqueeze(1)  # Shape: (num_embeddings, 1, embedding_dim)
    expanded_embeddings_2 = embeddings.unsqueeze(0)  # Shape: (1, num_embeddings, embedding_dim)

    # Calculate Euclidean distance along the last dimension (embedding_dim)
    distance_matrix = torch.norm(expanded_embeddings_1 - expanded_embeddings_2, dim=-1)
    return distance_matrix

# Convert to DataFrame for visualization
def similarity_matrix_to_dataframe(matrix):
    df = pd.DataFrame(matrix.numpy())
    return df

EMBEDS_PATH = '/home/renas/pythonprogv2/phd_xiaor_project/TSA_dataset/sim/poses/poses_2024-04-25_15-00-52_action_vocab.h5'

if __name__ == '__main__':
    embeddings, _ = load_embeddings(EMBEDS_PATH)
    similarity_matrix = compute_similarity_matrix(embeddings)
    print('cosine similarity matrix: ', similarity_matrix)
    similarity_df = similarity_matrix_to_dataframe(similarity_matrix)

    distance_matrix = compute_euclidean_distance_matrix(embeddings)
    distance_df = similarity_matrix_to_dataframe(distance_matrix)
    print('Eucledean distance matrix: ', distance_matrix)


    # Create a heatmap for the similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Cosine Similarity Matrix Heatmap')
    plt.xlabel('Embedding Index')
    plt.ylabel('Embedding Index')
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_df, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Euclidean Distance Matrix Heatmap')
    plt.xlabel('Embedding Index')
    plt.ylabel('Embedding Index')
    plt.show()