import torch
from torch_geometric.data import Data

def sample_negatives(pos_edge, num_nodes):
    pos_set = set((u.item(), v.item()) for u, v in pos_edge)
    neg_edges = []
    while len(neg_edges) < pos_edge.size(0):
        u = torch.randint(0, num_nodes, (1,)).item()
        v = torch.randint(0, num_nodes, (1,)).item()
        if (u, v) not in pos_set and (v, u) not in pos_set and u != v:
            neg_edges.append([u, v])
    return torch.tensor(neg_edges, dtype=torch.long)

def train_gnn(gnn, predictor, data, num_treatments, device, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(list(gnn.parameters()) + list(predictor.parameters()), lr=lr)
    loss_fn = torch.nn.BCELoss()
    pos_edge = data.edge_index.t()

    for epoch in range(epochs):
        gnn.train()
        predictor.train()
        optimizer.zero_grad()

        embeddings = gnn(data)
        pos_pred = predictor(embeddings[pos_edge[:, 0]], embeddings[pos_edge[:, 1]])
        pos_label = torch.ones(pos_pred.size(0), device=pos_pred.device)

        neg_edge = sample_negatives(pos_edge, num_treatments)
        neg_pred = predictor(embeddings[neg_edge[:, 0]], embeddings[neg_edge[:, 1]])
        neg_label = torch.zeros(neg_pred.size(0), device=neg_pred.device)

        loss = loss_fn(pos_pred, pos_label) + loss_fn(neg_pred, neg_label)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    gnn.eval()
    return gnn(data)
