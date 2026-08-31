
import torch
import torch.nn.functional as F

def data_no(data):
    min = data.min()
    max = data.max()
    data = (data - min) / (max - min)
    return data


def c(tensor):

    w = (tensor > 0.25).float()
    return w


def cosine_similarities(tensor):
    tensor_normalized = F.normalize(tensor, p=2, dim=-1)
    cos_similarities = torch.bmm(tensor_normalized, tensor_normalized.transpose(1, 2))
    return cos_similarities

def connections(input_tensor):


    threshold=0.041
    assert input_tensor.dim() == 2 and input_tensor.shape[0] == 1
    diff_matrix = torch.abs(input_tensor.unsqueeze(2) - input_tensor.unsqueeze(1))
    adjacency_matrix = (diff_matrix < threshold).float()
    return adjacency_matrix

