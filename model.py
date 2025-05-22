import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from use.mawno import MAWNONet
from use.config_l import Config
import random
from torch_geometric.nn import MessagePassing
import math
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops,degree
from use.seq_feature import ProteinFeatureExtractor

config = Config()
device = "cuda:1" if torch.cuda.is_available() else "cpu"
conv_filters = [[1,32],[3,32],[5,64],[7,128]]
embedding_size = output_dim = 256
d_ff = 256
n_heads = 8
d_k = 16
n_layer = 1
smi_vocab_size = 53
seq_vocab_size = 21
seed = 990721

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
np.random.seed(seed)

class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

class Smi_Encoder(nn.Module):
    def __init__(self):  
        super().__init__()
        self.config = config
        self.mawnonet = MAWNONet(config)

    def forward(self, smi_input):

        #  (batch_size, 256, length)
        if smi_input.dim() == 3 and smi_input.shape[1] == 256:
            smi_input_tensor = smi_input.permute(0, 2, 1).float()  
            smi_input_tensor = smi_input_tensor.to(self.config.device)
            output_emb = self.mawnonet(smi_input_tensor)
            return output_emb
        else:
            raise ValueError("Invalid input dimension for Smi_Encoder.")
#GRU\CNN
class Seq_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_feature_extractor = ProteinFeatureExtractor()
        self.graph_linear = nn.Linear(481, 256)
        self.seq_linear = nn.Linear(480, 256) 
        
        self.gru = nn.GRU(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu = nn.ReLU().to(device)

    def forward(self, protein_input):

        graph_embeddings = []
        for protein_data in protein_input:

            graph_data = protein_data[1]
            graph_embedding = self.graph_feature_extractor(graph_data)
            graph_embeddings.append(graph_embedding.x)

        graph_embeddings_tensor = torch.stack(graph_embeddings)
        graph_embedding_256d = self.graph_linear(graph_embeddings_tensor)
        

        # seq
        seq_embeddings_list = []
        for protein_data in protein_input:
  
            sequence_data = protein_data[1]
            token_representation = sequence_data["token_representation"]
            token_representation = token_representation.float()
            seq_embeddings_list.append(self.seq_linear(token_representation))
        seq_embeddings_tensor = torch.stack(seq_embeddings_list)
        # GRU
        gru_output, _ = self.gru(seq_embeddings_tensor)
        # CNN（N,C,L -> N,L,C）
        seq_embedding_for_cnn = gru_output.permute(0, 2, 1)
        cnn_output = self.conv(seq_embedding_for_cnn).to(device)
        cnn_output = self.relu(cnn_output)
        cnn_output = cnn_output.permute(0, 2, 1)#（N,L,C -> N,C,L）
        

        fused_embedding = torch.cat((cnn_output, graph_embedding_256d), dim=1)
        #fused_embedding = torch.cat((seq_embeddings_tensor,graph_embedding_256d),dim=1)

        return fused_embedding

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_encoder = Seq_Encoder()
        self.smi_encoder = Smi_Encoder()

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Squeeze(),
            nn.Linear(1024,256),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(256,64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64,1),
            Squeeze())

    def forward(self,seq_encode, smi_encode):
        seq_outputs = self.seq_encoder(seq_encode)
        smi_outputs = self.smi_encoder(smi_encode)

        seq_last_dim = seq_outputs.size(-1)
        smi_last_dim = smi_outputs.size(-1)

        seq_norm = nn.LayerNorm(seq_last_dim).to(device)
        smi_norm = nn.LayerNorm(smi_last_dim).to(device)
        seq_outputs = seq_norm(seq_outputs)
        smi_outputs = smi_norm(smi_outputs)
        score = torch.matmul(seq_outputs, smi_outputs.transpose(-1, -2))/np.sqrt(embedding_size)

        final_outputs = self.fc(score)
        return final_outputs
