import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.manifold import TSNE
import plotly.express as px
import pdb
import json


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets, vocab_size, subset_ratio=1.0, seed=42):
        assert 0 < subset_ratio <= 1.0, "subset_ratio must be in (0, 1]"
        
        indices = np.arange(len(sequences))
        if subset_ratio < 1.0:
            np.random.seed(seed)
            indices = np.random.choice(indices, size=int(len(indices) * subset_ratio), replace=False)

        self.sequences = pad_sequence([torch.tensor(sequences[i]) for i in indices], batch_first=True)
        self.targets = torch.stack([self.create_soft_label(targets[i], vocab_size) for i in indices])

    def create_soft_label(self, label_dict, vocab_size):
        label_vec = torch.zeros(vocab_size, dtype=torch.float32)
        for idx, weight in label_dict.items():
            label_vec[idx] = weight
        return label_vec

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class TransformerNextItemPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(512, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, vocab_size)
        )
        self.attention_weights = []

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        x = self.embedding(x) + self.positional_embedding(positions)
        out = self.transformer(x)
        last_hidden = out[:, -1, :]
        return self.fc(self.dropout(last_hidden))

        

class Next_Prediction_Recommender:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SAVE_PATH = 'results/Next_Prediction_Recommender/'

    def visualize_embeddings(self):
        self.model.eval()
        embeddings = self.model.embedding.weight.detach().cpu().numpy()
        item_names = [self.idx2item[i] for i in range(self.num_items)]

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embedding_2d = tsne.fit_transform(embeddings)

        df_plot = pd.DataFrame({
            'x': embedding_2d[:, 0],
            'y': embedding_2d[:, 1],
            'label': item_names
        })

        fig = px.scatter(df_plot, x='x', y='y', hover_name='label', title='t-SNE of Treatment Embeddings')
        fig.update_traces(marker=dict(size=8))
        fig.write_html("treatment_embeddings_visualization.html")

    def load_gnn_embeddings(self, path="./data/gnn_embeddings/treatment_embeddings_5L.json", freeze=True):
        with open(path, "r") as f:
            loaded_embeddings = json.load(f)

        with torch.no_grad():
            for item, emb in loaded_embeddings.items():
                try:
                    idx = int(self.item_encoder.transform(np.array([item])))
                    emb_tensor = torch.tensor(emb, dtype=torch.float, device=self.device)
                    self.model.embedding.weight[idx] = emb_tensor
                except Exception as e:
                    print(f"Warning: Could not load embedding for item {item}: {e}")

        if freeze:
            self.model.embedding.weight.requires_grad = False
            print(f"✅ Loaded {len(loaded_embeddings)} embeddings and **froze** item embedding.")
        else:
            print(f"✅ Loaded {len(loaded_embeddings)} embeddings (item embedding remains trainable).")
        
        return

    def freeze_treatment_embeddings(self):
        self.model.embedding.weight.requires_grad = False
        print(f"**Froze** item embedding.")
        return

    def user_item2idx(self, df, user_clm='user_id', item_clm='item'):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        df['user_idx'] = self.user_encoder.fit_transform(df[user_clm])
        df['item_idx'] = self.item_encoder.fit_transform(df[item_clm])

        self.num_users = df['user_idx'].nunique()
        self.num_items = df['item_idx'].nunique()
        self.user_clm = user_clm
        self.item_clm = item_clm
        self.user2idx = dict(zip(self.user_encoder.classes_, range(len(self.user_encoder.classes_))))
        self.idx2item = dict(enumerate(self.item_encoder.classes_))
        self.item2idx = {v: k for k, v in self.idx2item.items()}
        return df


    def build_sequences(self, df):
        user_seqs = df.groupby('user_idx')['item_idx'].apply(list)
        max_len = self.args.MAX_SEQ_LEN
        decay = self.args.DECAY_ALPHA
        X, y = [], []
        for seq in user_seqs:
            for i in range(1, len(seq)):
                prefix = seq[:i]
                if len(prefix) > max_len:
                    prefix = prefix[-max_len:]
                soft_label = {}
                for j, idx in enumerate(seq[i:]):
                    weight = decay ** j
                    soft_label[idx] = soft_label.get(idx, 0) + weight
                y_sum = sum(soft_label.values())
                for k in soft_label:
                    soft_label[k] /= y_sum
                X.append(prefix)
                y.append(soft_label)
        return X, y

    def train(self, train_df):
        self.train_df = train_df
        X, y = self.build_sequences(train_df)
        dataset = SequenceDataset(X, y, self.num_items)

        loader = DataLoader(dataset, batch_size=self.args.BATCH_SIZE, shuffle=True)

        self.model = TransformerNextItemPredictor(
            self.num_items,
            embed_dim=self.args.EMBED_DIM,
            num_heads=self.args.NUM_HEADS,
            num_layers=self.args.NUM_LAYERS,
            dropout=self.args.DROPOUT
        )

        # self.load_gnn_embeddings(freeze=True)
        self.freeze_treatment_embeddings()

        self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.LR)
        loss_fn = nn.CrossEntropyLoss() # label_smoothing=0.1

        self.model.train()
        for epoch in range(self.args.NUM_EPOCHS):
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                # pred = self.model(batch_x)
                # loss = loss_fn(pred, batch_y)

                logits = self.model(batch_x)
                log_probs = F.log_softmax(logits, dim=-1)
                loss = F.kl_div(log_probs, batch_y, reduction='batchmean')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.args.NUM_EPOCHS} - Loss: {total_loss:.4f}, Normalized: {total_loss/len(loader):.4f}")

    def recommend(self, history_list, top_k=5):
        self.model.eval()
        history_idx_list = [self.item2idx[treat] for treat in history_list]

        seq_tensor = pad_sequence([torch.tensor(history_idx_list)], batch_first=True).to(self.device)
        with torch.no_grad():
            scores = self.model(seq_tensor)
            probs = torch.softmax(scores, dim=-1).squeeze().cpu().numpy()

        top_k_indices = np.argsort(probs)[-top_k:][::-1]
        top_k_names = self.item_encoder.inverse_transform(top_k_indices)
        top_k_scores = probs[top_k_indices]

        return list(zip(top_k_names, top_k_scores))

    def evaluate(self, val_df, top_k, save_log=False):
        train_histories = self.train_df.groupby('user_idx')['item_idx'].apply(list).to_dict()

        original_val_len = len(val_df)
        val_df = val_df[val_df[self.user_clm].isin(self.user2idx) & val_df[self.item_clm].isin(self.item_encoder.classes_)].copy()
        filtered_val_len = len(val_df)
        dropped_val_len = original_val_len - filtered_val_len

        print(f"Validation interactions (after filtering): {filtered_val_len}")
        print(f"Dropped validation interactions: {dropped_val_len/original_val_len:.1%}")

        val_df['user_idx'] = self.user_encoder.transform(val_df[self.user_clm])
        val_df['item_idx'] = self.item_encoder.transform(val_df[self.item_clm])

        recall_total, precision_total, hit_total, mrr_total, total_users = 0, 0, 0, 0, 0

        for user_idx, group in val_df.groupby('user_idx'):
            history = train_histories.get(user_idx, [])
            if not history:
                continue

            true_items = set(group[self.item_clm])
            true_count = len(true_items)
            if true_count == 0:
                continue

            if len(history) > self.args.MAX_SEQ_LEN:
                history = history[-self.args.MAX_SEQ_LEN:]

            input_tensor = pad_sequence([torch.tensor(history)], batch_first=True).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                scores = self.model(input_tensor)
                top_k_items = torch.topk(scores, top_k).indices.cpu().numpy()
                top_k_items = np.atleast_1d(top_k_items.squeeze()).tolist()
                recommended_k = [self.idx2item[i] for i in top_k_items]

            hits_k = set(recommended_k) & true_items

            recall_total += len(hits_k) / true_count
            precision_total += len(hits_k) / top_k
            hit_total += 1 if hits_k else 0

            for rank, item in enumerate(recommended_k, start=1):
                if item in true_items:
                    mrr_total += 1 / rank
                    break

            total_users += 1

        recall = recall_total / total_users if total_users > 0 else 0
        precision = precision_total / total_users if total_users > 0 else 0
        hit_rate = hit_total / total_users if total_users > 0 else 0
        mrr = mrr_total / total_users if total_users > 0 else 0

        if save_log:
            os.makedirs(self.SAVE_PATH, exist_ok=True)
            with open(self.SAVE_PATH + f"Next_Prediction_Recommender_top{top_k}.txt", "w") as f:
                def log(msg):
                    print(msg)
                    f.write(msg + "\n")
                log(f"🎯 Recall@{top_k}:         {recall:.4f}")
                log(f"📐 Precision@{top_k}:      {precision:.4f}")
                log(f"✅ Hit Rate@{top_k}:       {hit_rate:.4f}")
                log(f"📈 MRR@{top_k}:            {mrr:.4f}")
        else:

            print(f"🎯 Recall@{top_k}:         {recall:.4f}")
            print(f"📐 Precision@{top_k}:      {precision:.4f}")
            print(f"✅ Hit Rate@{top_k}:       {hit_rate:.4f}")
            print(f"📈 MRR@{top_k}:            {mrr:.4f}")

        # self.visualize_embeddings()
