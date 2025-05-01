import torch
from torch_geometric.data import Data
import pandas as pd
import pdb
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # if using GPU
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

DATA_PATH = '../data/train_df.csv'
train_df = pd.read_csv(DATA_PATH)

patient_records = dict()

for user_id, group in train_df.groupby('user_id'):
    # pdb.set_trace()
    patient_records.update({user_id: group['item'].to_list()})
    # patient_records.update({user_id: group.sort_values(by=['date'])['item'].to_list()})


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# patient_records = {
#     "patient1": ["A", "B", "C"],
#     "patient2": ["A", "B"],
#     "patient3": ["B", "C"],
#     "patient4": ["D", "E"],
# }

from collections import Counter
from itertools import combinations


cooccurrence_counter = Counter()

for treatments in patient_records.values():
    for t1, t2 in combinations(sorted(treatments), 2):
        cooccurrence_counter[(t1, t2)] += 1
        cooccurrence_counter[(t2, t1)] += 1


# for treatments in patient_records.values():
#     for idx1 in range(0, len(treatments)-1):
#         for idx2 in range(idx1+1, len(treatments)):
#             cooccurrence_counter[(treatments[idx1], treatments[idx2])] += 1


treatments = set()
for record in patient_records.values():
    treatments.update(record)

treatment2id = {treatment: idx for idx, treatment in enumerate(sorted(treatments))}
id2treatment = {idx: treatment for treatment, idx in treatment2id.items()}
num_treatments = len(treatment2id)


import torch


edges = []
weights = []

for (t1, t2), count in cooccurrence_counter.items():
    src = treatment2id[t1]
    dst = treatment2id[t2]
    edges.append([src, dst])
    weights.append(count)



edge_index = torch.tensor(edges, dtype=torch.long).t()
edge_weight = torch.tensor(weights, dtype=torch.float)


import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv


class TreatmentGNN(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embed_dim)
        self.node_embeddings.weight.requires_grad = False
        # self.scale = nn.Parameter(torch.tensor(100.0))
        self.conv1 = GCNConv(embed_dim, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 256)
        self.conv4 = GCNConv(256, 128)
        self.conv5 = GCNConv(128, embed_dim)
        # pdb.set_trace()


    def forward(self, data):
        x = self.node_embeddings.weight
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = self.conv4(x, edge_index, edge_weight).relu()
        x = self.conv5(x, edge_index, edge_weight)

        # x = F.normalize(x, p=2, dim=1)
        # x = F.normalize(x, p=2, dim=1) * self.scale

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




# Create graph data
data = Data(edge_index=edge_index, edge_weight=edge_weight).to(device)

# Models
embed_dim = 128
gnn = TreatmentGNN(num_nodes=num_treatments, embed_dim=embed_dim).to(device)

predictor = LinkPredictor(embed_dim=embed_dim).to(device)

optimizer = torch.optim.Adam(list(gnn.parameters()) + list(predictor.parameters()), lr=0.01)
loss_fn = nn.BCELoss()


# Positive edges are just the existing edges
pos_edge = edge_index.t()  # Shape [num_edges, 2]

# Function to generate true negative edges
def sample_negatives(pos_edge, num_nodes):
    pos_set = set((u.item(), v.item()) for u, v in pos_edge)
    neg_edges = []
    while len(neg_edges) < pos_edge.size(0):
        u = torch.randint(0, num_nodes, (1,)).item()
        v = torch.randint(0, num_nodes, (1,)).item()
        if (u, v) not in pos_set and (v, u) not in pos_set and u != v:
            neg_edges.append([u, v])
    return torch.tensor(neg_edges, dtype=torch.long)


for epoch in range(300):
    gnn.train()
    predictor.train()
    optimizer.zero_grad()

    embeddings = gnn(data)

    pos_pred = predictor(embeddings[pos_edge[:, 0]], embeddings[pos_edge[:, 1]])
    pos_label = torch.ones(pos_pred.size(0), device=pos_pred.device)

    # Negative samples
    neg_edge = sample_negatives(pos_edge, num_treatments)
    neg_pred = predictor(embeddings[neg_edge[:, 0]], embeddings[neg_edge[:, 1]])
    neg_label = torch.zeros(neg_pred.size(0), device=neg_pred.device)

    # Loss
    loss = loss_fn(pos_pred, pos_label) + loss_fn(neg_pred, neg_label)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


gnn.eval()
final_treatment_embeddings = gnn(data)
print(final_treatment_embeddings)


treatment_embeddings_np = final_treatment_embeddings.detach().cpu().numpy()

# Create dictionary {item: embedding list}
treatment_embedding_dict = {
    id2treatment[idx]: treatment_embeddings_np[idx].tolist()
    for idx in range(num_treatments)
}

import json

with open("treatment_embeddings.json", "w") as f:
    json.dump(treatment_embedding_dict, f)

print("Successfully saved treatment embeddings.")

from sklearn.manifold import TSNE

# Move embeddings to CPU and detach
embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(final_treatment_embeddings.detach().cpu().numpy())

import plotly.express as px

# Create a list of treatment names for hovering
hover_labels = [id2treatment[idx] for idx in range(num_treatments)]

# Create Plotly scatter plot
fig = px.scatter(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    hover_name=hover_labels,
    title="Treatment Embedding Visualization (t-SNE)",
    width=800,
    height=600
)

fig.update_traces(marker=dict(size=8))

# Save to HTML
fig.write_html("treatment_embeddings_visualization.html")