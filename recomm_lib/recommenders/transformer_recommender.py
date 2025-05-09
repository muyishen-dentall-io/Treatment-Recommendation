import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import json
from torch.optim.lr_scheduler import CosineAnnealingLR

class SequenceDataset(Dataset):
    def __init__(self, sequences, targets, vocab_size, subset_ratio=1.0, seed=42):
        assert 0 < subset_ratio <= 1.0, "subset_ratio must be in (0, 1]"

        indices = np.arange(len(sequences))
        if subset_ratio < 1.0:
            np.random.seed(seed)
            indices = np.random.choice(indices, size=int(len(sequences) * subset_ratio), replace=False)

        self.sequences = pad_sequence([torch.tensor(sequences[i], dtype=torch.long) for i in indices], batch_first=True)
        self.targets = torch.stack([self.create_label(targets[i], vocab_size) for i in indices])
        self.vocab_size = vocab_size

    def create_label(self, target_set, vocab_size):
        label_vec = torch.zeros(vocab_size, dtype=torch.float32)
        weight = 1.0 / len(target_set)
        for idx in target_set:
            label_vec[idx] = weight
        return label_vec

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# Transformer model
class TransformerStatusEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.encoder(x)
        return self.dropout(x[:, -1, :])  # use last token as status embedding

# Loss
def multi_positive_infonce_loss(status_emb, label_distributions, item_embeddings, temperature=0.07):
    sim_logits = torch.matmul(status_emb, item_embeddings.T) / temperature
    log_probs = F.log_softmax(sim_logits, dim=1)
    loss = -torch.sum(label_distributions * log_probs) / status_emb.size(0)
    return loss

def cosine_anneal_temperature(epoch, total_epochs, min_temp=0.05, max_temp=0.15):
        ratio = epoch / total_epochs
        return min_temp + 0.5 * (max_temp - min_temp) * (1 + np.cos(np.pi * ratio))

# Recommender system
class TransformerRecommender:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SAVE_PATH = 'results/Transformer_Recommender/'

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

    def build_sequences(self, df):
        user_seqs = df.groupby('user_idx')['item_idx'].apply(list)
        max_len = self.args.MAX_SEQ_LEN
        X, y = [], []
        for seq in user_seqs:
            for i in range(1, len(seq)):
                prefix = seq[:i]
                if len(prefix) > max_len:
                    prefix = prefix[-max_len:]
                future_items = set(seq[i:])  # no decay
                X.append(prefix)
                y.append(future_items)
        return X, y
    
    

    def train(self, train_df):
        self.train_df = train_df
        X, y = self.build_sequences(train_df)
        dataset = SequenceDataset(X, y, self.num_items)
        loader = DataLoader(dataset, batch_size=self.args.BATCH_SIZE, shuffle=True)

        self.model = TransformerStatusEncoder(
            vocab_size=self.num_items,
            embed_dim=self.args.EMBED_DIM,
            num_heads=self.args.NUM_HEADS,
            num_layers=self.args.NUM_LAYERS,
            dropout=self.args.DROPOUT,
            max_len=self.args.MAX_SEQ_LEN
        ).to(self.device)

        # self.load_gnn_embeddings(freeze=False)
        self.freeze_treatment_embeddings()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.LR)
        # scheduler = CosineAnnealingLR(optimizer, T_max=self.args.NUM_EPOCHS)

        self.model.train()
        for epoch in range(self.args.NUM_EPOCHS):
            current_temp = cosine_anneal_temperature(epoch, self.args.NUM_EPOCHS)
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                status_emb = self.model(batch_x)
                # loss = multi_positive_infonce_loss(status_emb, batch_y, self.model.embedding.weight)
                loss = multi_positive_infonce_loss(status_emb, batch_y, self.model.embedding.weight, temperature=current_temp)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.args.NUM_EPOCHS} - Loss: {total_loss/len(loader):.4f}")

    def evaluate(self, val_df, top_k=5):
        train_histories = self.train_df.groupby('user_idx')['item_idx'].apply(list).to_dict()

        original_val_len = len(val_df)
        val_df = val_df[val_df[self.user_clm].isin(self.user2idx) & val_df[self.item_clm].isin(self.item_encoder.classes_)].copy()
        filtered_val_len = len(val_df)
        dropped_val_len = original_val_len - filtered_val_len

        print(f"✅ Validation interactions (after filtering): {filtered_val_len}")
        print(f"⚠️ Dropped validation interactions: {dropped_val_len/original_val_len:.1%}")

        val_df['user_idx'] = self.user_encoder.transform(val_df[self.user_clm])
        val_df['item_idx'] = self.item_encoder.transform(val_df[self.item_clm])

        recall_total, precision_total, hit_total, mrr_total, total_users = 0, 0, 0, 0, 0

        self.model.eval()
        with torch.no_grad():
            for user_idx, group in val_df.groupby('user_idx'):
                history = train_histories.get(user_idx, [])
                if len(history) == 0:
                    continue

                if len(history) > self.args.MAX_SEQ_LEN:
                    history = history[-self.args.MAX_SEQ_LEN:]

                input_tensor = pad_sequence([torch.tensor(history)], batch_first=True).to(self.device)
                status_emb = self.model(input_tensor)  # shape: [1, D]
                item_embs = self.model.embedding.weight  # shape: [num_items, D]

                scores = F.cosine_similarity(status_emb, item_embs.unsqueeze(0), dim=-1).squeeze()  # [num_items]
                topk_indices = torch.topk(scores, top_k).indices.cpu().tolist()
                recommended_items = [self.idx2item[i] for i in topk_indices]

                true_items = set(group[self.item_clm])
                hits_k = set(recommended_items) & true_items

                recall_total += len(hits_k) / len(true_items)
                precision_total += len(hits_k) / top_k
                hit_total += 1 if hits_k else 0

                for rank, item in enumerate(recommended_items, start=1):
                    if item in true_items:
                        mrr_total += 1 / rank
                        break

                total_users += 1

        recall = recall_total / total_users if total_users > 0 else 0
        precision = precision_total / total_users if total_users > 0 else 0
        hit_rate = hit_total / total_users if total_users > 0 else 0
        mrr = mrr_total / total_users if total_users > 0 else 0

        os.makedirs(self.SAVE_PATH, exist_ok=True)
        with open(self.SAVE_PATH + f"Transformer_Recommender_top{top_k}.txt", "w") as f:
            def log(msg):
                print(msg)
                f.write(msg + "\n")
            log(f"🎯 Recall@{top_k}:         {recall:.4f}")
            log(f"📐 Precision@{top_k}:      {precision:.4f}")
            log(f"✅ Hit Rate@{top_k}:       {hit_rate:.4f}")
            log(f"📈 MRR@{top_k}:            {mrr:.4f}")