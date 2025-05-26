import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch

class TreatmentGNN(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embed_dim)
        self.node_embeddings.weight.requires_grad = False
        self.conv1 = GCNConv(embed_dim, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 256)
        self.conv4 = GCNConv(256, 128)
        self.conv5 = GCNConv(128, embed_dim)

    def forward(self, data):
        x = self.node_embeddings.weight
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = self.conv4(x, edge_index, edge_weight).relu()
        x = self.conv5(x, edge_index, edge_weight)
        return x

class LinkPredictor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, emb_src, emb_dst):
        x = torch.cat([emb_src, emb_dst], dim=1)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze()
