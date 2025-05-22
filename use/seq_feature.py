import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, knn_graph, GATConv# 导入GATConv
from torch_geometric.utils import to_undirected, add_remaining_self_loops, to_dense_batch, degree
from torch_geometric.data import Data
from torch_scatter import scatter_mean,scatter_softmax
from torch_geometric.nn import TopKPooling, global_mean_pool
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
# GATConv
class FirstGeoConvBlock(nn.Module):
    def __init__(self, gcn, in_channels, out_channels, heads=4, dropout=0.6, concat=True, negative_slope=0.2, **kwargs):
        super(FirstGeoConvBlock, self).__init__()
        self.conv1 = gcn(in_channels, out_channels, heads=heads, dropout=dropout, concat=concat, negative_slope=negative_slope, **kwargs)
        #self.bn1 = nn.BatchNorm1d(out_channels if concat else out_channels * heads)
        self.bn1 = nn.LayerNorm(2048)
        self.relu1 = nn.ReLU()

        self.conv2 = gcn(2048, 481, heads=3, dropout=dropout, concat=True, negative_slope=negative_slope, **kwargs)
        self.bn2 = nn.LayerNorm(1443)
        self.relu2 = nn.ReLU()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # GAT1
        x=x.float()
        x = self.conv1(x, edge_index, edge_attr)#[1098,481]
        x = self.bn1(x)
        x = self.relu1(x)#[1-89,2048]

        # GAT2
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = self.relu2(x)

        data.x = x
        return data

class HierarchicalPoolingLayer(nn.Module):
    def __init__(self, num_layers, pool_ratios):
        super(HierarchicalPoolingLayer, self).__init__()
        self.num_layers = num_layers
        self.pooling_layers = nn.ModuleList()

        # Initialize pooling layers with the correct input feature dimension
        for layer_idx in range(self.num_layers):
            self.pooling_layers.append(TopKPooling(1443, pool_ratios[layer_idx]))  # 481 features per node

    def forward(self, data):
        x = data.x  # Node features of shape [num_nodes, 481]
        edge_index = data.edge_index  # Shape [2, e]
        edge_weight = data.edge_attr  # Shape [e]

        # Ensure dimensions are correct
        assert x.shape[1] == 1443, "Feature dimension of x must be 481"
        assert edge_index.size(0) == 2, "edge_index must have 2 rows"
        if edge_weight.size(0)!= edge_index.size(1):
            raise ValueError("Edge weight and edge index dimensions do not match. Edge weight size: {}, Edge index size: {}. Check the graph construction or previous operations.".format(edge_weight.size(0), edge_index.size(1)))

        for layer in self.pooling_layers:
            score = edge_weight

            # Perform pooling
            outputs = layer(x, edge_index)
            # Unpack according to the expected number of outputs
            x = outputs[0]  
            edge_index = outputs[1]  
            # outputs[2] is None - no edge attributes, ignore it
            batch = outputs[3] if outputs[3] is not None else None  
            new_edge_weights = outputs[4]  
            additional_scores = outputs[5]  
            if new_edge_weights is not None:
                new_edge_weights = new_edge_weights[:edge_index.size(1)]  

        # Global pooling to summarize the graph
        pooled_x = global_mean_pool(x, batch)
        data.x = pooled_x  # Assign pooled features back to data
        return data


# graph data feature extractor
class ProteinFeatureExtractor(nn.Module):
    def __init__(self, hidden_channels=64, out_channels=32, transform=None):
        super(ProteinFeatureExtractor, self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.transform = transform

        self.gcn = GATConv
        self.gcn_kwargs = {
            'in_channels': 481,  
            'out_channels': 256,
            'heads': 8,  
            'dropout': 0.5,  
            'concat': True, 
            'negative_slope': 0.2  
        }

        # GATConv
        self.geo_conv_block = FirstGeoConvBlock(self.gcn, **self.gcn_kwargs)
	#Top-K Pooling
        num_layers = 3
        pool_ratios = [0.5, 0.3, 0.2]
        self.pool_layer = HierarchicalPoolingLayer(num_layers, pool_ratios)

        self.norm_layer = nn.BatchNorm1d(1443)
        self.ln =nn.LayerNorm(481)

        self.out_linear = nn.Linear(1443, 481)

    def forward(self, protein_data):
        token_representation = protein_data["token_representation"]
        num_pos = protein_data["num_pos"]
        edge_index = protein_data["edge_index"]
        edge_weight = protein_data["edge_weight"]
        # cat
        x = torch.cat([token_representation, num_pos], dim=1)
        graph_data_tg = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x = x.float()
        x = self.ln(x)

        graph_data_tg = self.geo_conv_block(graph_data_tg)
        x = graph_data_tg.x
   
        x = self.norm_layer(x)


        graph_data_tg = self.pool_layer(graph_data_tg)
        x = graph_data_tg.x


        x = self.out_linear(x)
        if self.transform is not None:
            graph_data_tg.x = self.transform(graph_data_tg.x)
        graph_data_tg.x = x
        return graph_data_tg




