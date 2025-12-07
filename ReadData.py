import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from PIL import ImageFile
from scipy.io import loadmat
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler

def log_zscore_normalize(data):
    data = np.sign(data) * np.log1p(np.abs(data))
    mean = np.mean(data, axis=(1, 2), keepdims=True)
    std = np.std(data, axis=(1, 2), keepdims=True) + 1e-6
    return (data - mean) / std

def normalize(data):
    min_vals = np.min(data, axis=(1, 2), keepdims=True)
    max_vals = np.max(data, axis=(1, 2), keepdims=True)
    normalized_data = (data - min_vals) / (max_vals - min_vals +1e-5)
    return normalized_data

def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x

def ADNI_DTI(args):
    # load fMRI
    m = loadmat('dataset/...')
    data = m['feas']  # (306, 90, 197)
    labels = m['label'][0]  # (306, )

    start = (197 - 195) // 2
    end = start + 196
    data = data[:, :, start:end]  # (306, 90, 196)

    # load DTI
    n = loadmat('dataset/...')  # structural brain connectivity
    dti_data = n['DTI']
    if dti_data.shape[0] == 90 and dti_data.shape[2] == data.shape[0]:
        dti_data = dti_data.transpose(2, 0, 1)  # -> (306, 90, 90)

    # ----- Label filtering section (uncomment as needed for the task) -----

    for i in range(labels.shape[0]):
        if labels[i] == 2:
            labels[i] = 1

    # select labels 0 and 1
    # bool_idx = (labels == 0) | (labels == 1)
    # data = data[bool_idx]
    # dti_data = dti_data[bool_idx]
    # labels = labels[bool_idx]

    # select labels 0 and 2
    # bool_idx = (labels == 0) | (labels == 2)
    # data = data[bool_idx]
    # dti_data = dti_data[bool_idx]
    # labels = labels[bool_idx]
    # for i in range(labels.shape[0]):
    #     if labels[i] == 2:
    #         labels[i] = 1

    # select labels 1 and 2
    # bool_idx = (labels == 1) | (labels == 2)
    # data = data[bool_idx]
    # dti_data = dti_data[bool_idx]
    # labels = labels[bool_idx]
    # for i in range(labels.shape[0]):
    #     if labels[i] == 1:
    #         labels[i] = 0
    #     if labels[i] == 2:
    #         labels[i] = 1

    # -------------------------------------------------------------------

    data = normalize(data)
    dti_data = normalize(dti_data)

    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    data = data[index]
    dti_data = dti_data[index]
    labels = labels[index]

    #PyTorch Tensor
    data_tensor = torch.from_numpy(data).float()       # (B, 90, 196)
    dti_data = dti_data.astype(np.float32)
    dti_tensor = torch.from_numpy(dti_data)
    labels_tensor = torch.from_numpy(labels).long()    # (B,)

    num_nodes = data_tensor.size(1)
    seq_length = data_tensor.size(2)
    num_classes = torch.unique(labels_tensor).size(0)

    dataset = TensorDataset(data_tensor, dti_tensor, labels_tensor)
    return dataset, num_nodes, seq_length, num_classes

