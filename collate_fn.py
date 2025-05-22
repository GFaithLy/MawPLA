from torch.utils.data import DataLoader
import torch
import numpy as np
from torch import nn, optim
from model import MyModule
from Dataset import MyDataset
import sklearn.metrics as m
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from io import BytesIO
from IPython.display import Image
import numpy as np
import torch
from torch_geometric.utils import to_undirected
device = "cuda:1" if torch.cuda.is_available() else "cpu"

def custom_collate_fn(batch):
    id_names, smi_encodes, seq_encodes, affinities = zip(*batch)


    max_num_nodes = 0
    max_num_edges = 0
    for seq_encode in seq_encodes:
        num_nodes = seq_encode.graph["num_nodes"]
        edge_index = seq_encode.graph["edge_index"]
        max_num_nodes = max(max_num_nodes, num_nodes)
        max_num_edges = max(max_num_edges, edge_index.shape[1])

    new_seq_encodes = []
    for seq_encode in seq_encodes:
        num_nodes = seq_encode.graph["num_nodes"]
        edge_index = seq_encode.graph["edge_index"]
        edge_weight = seq_encode.graph["edge_weight"]

        # 检查 token_representation 是否存在
        if "token_representation" not in seq_encode.graph:
            raise KeyError("token_representation not found in seq_encode.graph")

        token_representation = seq_encode.graph["token_representation"].to(device)

        # 处理节点数
        pad_nodes = max_num_nodes - num_nodes
        if pad_nodes > 0:
            new_token_representation = torch.cat((
                token_representation,
                torch.zeros((pad_nodes, token_representation.shape[1]), device=device)
            ), dim=0)
            new_num_pos = torch.cat((
                seq_encode.graph["num_pos"].to(device),
                torch.zeros((pad_nodes, 1), device=device)
            ), dim=0)
        else:
            new_token_representation = token_representation[:max_num_nodes, :]
            new_num_pos = seq_encode.graph["num_pos"][:max_num_nodes, :].to(device)

        # 处理边数
        pad_edges = max_num_edges - edge_index.shape[1]
        if pad_edges > 0:

            new_edge_index = torch.cat((
                edge_index.to(device),
                torch.zeros((2, pad_edges), dtype=edge_index.dtype, device=device)
            ), dim=1)
            new_edge_weight = torch.cat((
                edge_weight.to(device),
                torch.zeros(pad_edges, device=device)
            ), dim=0)
        else:
            new_edge_index = edge_index[:, :max_num_edges].to(device)
            new_edge_weight = edge_weight[:max_num_edges].to(device)

        new_graph = {
            "num_nodes": max_num_nodes,
            "token_representation": new_token_representation,
            "num_pos": new_num_pos,
            "edge_index": new_edge_index,
            "edge_weight": new_edge_weight,
        }

        new_sequence = {
            #"seq": seq_encode.sequence["seq"],
            #"seq_feat": seq_encode.sequence["seq_feat"].to(device)
        }

        new_seq_encodes.append((new_sequence, new_graph))

    # 处理 smi_encodes 和 affinities
    new_smi_encodes = [torch.tensor(smi_encode, device=device).float() for smi_encode in smi_encodes]
    new_affinities = torch.tensor(affinities, dtype=torch.float32, device=device)

    if isinstance(id_names[0], str):
        new_id_names = list(id_names)
    elif isinstance(id_names[0], (int, float)):
        new_id_names = [type(id_names[0])(i) for i in id_names]
    else:
        raise TypeError(f"Unexpected type for id_names: {type(id_names[0])}")
    new_smi_encodes = torch.stack([torch.tensor(smi_encode, device=device).float() for smi_encode in smi_encodes])

    return new_id_names, new_smi_encodes, new_seq_encodes, new_affinities