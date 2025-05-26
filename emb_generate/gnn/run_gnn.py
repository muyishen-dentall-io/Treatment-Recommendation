from gnn_lib.utils import set_seed
from gnn_lib.dataset import load_patient_records, build_cooccurrence_edges, build_vocab
from gnn_lib.model import TreatmentGNN, LinkPredictor
from gnn_lib.train import train_gnn
from gnn_lib.visualize import plot_cooccurrence_graph, plot_treatment_embeddings_tsne
import torch
from torch_geometric.data import Data
import json

def main():
    set_seed(42)
    DATA_PATH = '../../data/train_df.csv'
    patient_records = load_patient_records(DATA_PATH)
    cooccurrence_counter = build_cooccurrence_edges(patient_records)
    treatment2id, id2treatment = build_vocab(patient_records)
    num_treatments = len(treatment2id)
    edges, weights = [], []

    for (t1, t2), count in cooccurrence_counter.items():
        src = treatment2id[t1]
        dst = treatment2id[t2]
        edges.append([src, dst])
        weights.append(count)

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = Data(edge_index=edge_index, edge_weight=edge_weight).to(device)
    embed_dim = 128
    gnn = TreatmentGNN(num_nodes=num_treatments, embed_dim=embed_dim).to(device)
    predictor = LinkPredictor(embed_dim=embed_dim).to(device)

    final_treatment_embeddings = train_gnn(gnn, predictor, data, num_treatments, device, epochs=200, lr=0.01)

    treatment_embeddings_np = final_treatment_embeddings.detach().cpu().numpy()
    treatment_embedding_dict = {
        id2treatment[idx]: treatment_embeddings_np[idx].tolist()
        for idx in range(num_treatments)
    }

    with open("treatment_embeddings.json", "w") as f:
        json.dump(treatment_embedding_dict, f)
    print("Successfully saved treatment embeddings.")

    plot_cooccurrence_graph(id2treatment, treatment2id, cooccurrence_counter)
    plot_treatment_embeddings_tsne(final_treatment_embeddings, id2treatment)

if __name__ == '__main__':
    main()
