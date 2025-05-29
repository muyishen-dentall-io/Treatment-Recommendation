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
    def __init__(self, sequences, targets, vocab_size, allowed_indices, max_len=512, subset_ratio=1.0, seed=42):
        assert 0 < subset_ratio <= 1.0, "subset_ratio must be in (0, 1]"

        indices = np.arange(len(sequences))
        if subset_ratio < 1.0:
            np.random.seed(seed)
            indices = np.random.choice(indices, size=int(len(sequences) * subset_ratio), replace=False)

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.sequences = torch.stack([self.pad_and_truncate(sequences[i]) for i in indices])
        self.targets = torch.stack([self.create_label(targets[i], vocab_size, allowed_indices) for i in indices])

    def pad_and_truncate(self, seq):
        seq = torch.tensor(seq, dtype=torch.long)
        if len(seq) < self.max_len:
            pad_len = self.max_len - len(seq)
            return F.pad(seq, (0, pad_len), value=0)
        else:
            return seq[-self.max_len:]

    def create_label(self, target_set, vocab_size, allowed_indices):
        label_vec = torch.zeros(vocab_size, dtype=torch.float32)
        filtered_targets = [idx for idx in target_set if idx in allowed_indices]
        if filtered_targets:
            weight = 1.0 / len(filtered_targets)
            for idx in filtered_targets:
                label_vec[idx] = weight
        return label_vec

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def pad_and_truncate(seq, max_len, pad_value=0):
    seq = torch.tensor(seq, dtype=torch.long)
    if len(seq) < max_len:
        pad_len = max_len - len(seq)
        return F.pad(seq, (0, pad_len), value=pad_value)
    else:
        return seq[-max_len:]

class TransformerStatusEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1, max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.embedding_bank = None
        self.use_avg_embedding = False
        self.norm = nn.LayerNorm(embed_dim)

    def set_embedding_bank(self, embedding_bank, use_avg=False, n_desc=5):
        self.n_desc = n_desc
        self.embedding_bank = embedding_bank
        self.use_avg_embedding = use_avg

    def forward(self, x):
        device = x.device
        B, T = x.size()

        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)

        if self.embedding_bank is None:
            raise ValueError("embedding_bank not set!")

        if self.use_avg_embedding:
            item_embs = self.embedding_bank.mean(dim=1)
            x_embed = item_embs[x]
        else:
            rand_indices = torch.randint(0, self.n_desc, (B, T), device=device)
            x_embed = self.embedding_bank[x, rand_indices]

        pos_embed = self.pos_embedding(positions)
        x = x_embed + pos_embed
        x = self.encoder(x)
        return self.dropout(x[:, -1, :])
    

def multi_positive_infonce_loss(status_emb, label_distributions, item_embeddings, allowed_indices, temperature=0.07):
    selected_embeddings = item_embeddings[allowed_indices]
    sim_logits = torch.matmul(status_emb, selected_embeddings.T) / temperature
    log_probs = F.log_softmax(sim_logits, dim=1)
    masked_labels = label_distributions[:, allowed_indices]
    loss = -torch.sum(masked_labels * log_probs) / status_emb.size(0)
    return loss

def cosine_anneal_temperature(epoch, total_epochs, min_temp=0.05, max_temp=0.2):
    ratio = epoch / total_epochs
    return min_temp + 0.5 * (max_temp - min_temp) * (1 + np.cos(np.pi * ratio))

class Status_Recommender:
    def __init__(self, args, item_list=[], validation_item_list=[]):
        self.args = args
        self.args.EVAL_TRAIN_EVERY = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SAVE_PATH = 'results/Status_Recommender/'

        self.item_list = item_list
        self.validation_item_list = validation_item_list
        self.allowed_items = sorted(set(item_list) - set(validation_item_list))

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

        if len(self.allowed_items) == 0:
            self.allowed_items = df['item'].unique().tolist()
            self.validation_item_list = df['item'].unique().tolist()

        self.allowed_item_indices = [self.item_encoder.transform([item])[0] for item in self.allowed_items]
        self.validation_idx = torch.tensor([self.item_encoder.transform([item])[0] for item in self.validation_item_list]).to(self.device)
        
        return df

    def load_llm_embeddings(self, path="./emb_generate/llm/description_embeddings.json", freeze=True):
        with open(path, "r") as f:
            raw_embeddings = json.load(f)

        treatment_name_list =  list(raw_embeddings.keys())

        if treatment_name_list:
            self.n_desc = len(raw_embeddings[treatment_name_list[0]])
        
        else:
            self.n_desc = 1

        self.embedding_bank = torch.zeros((self.num_items, self.n_desc, self.args.EMBED_DIM), device=self.device)
        for item, emb_list in raw_embeddings.items():
            try:
                idx = int(self.item_encoder.transform([item])[0])
                for j in range(min(self.n_desc, len(emb_list))):
                    self.embedding_bank[idx, j] = torch.tensor(emb_list[j], dtype=torch.float)
            except Exception as e:
                print(f"Warning: Could not load embedding for item {item}: {e}")

        if freeze:
            self.embedding_bank.requires_grad = False
            print(f"Loaded {len(raw_embeddings)} items with {self.n_desc} embeddings each. Bank frozen.")
        
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
                future_items = set(seq[i:])
                X.append(prefix)
                y.append(future_items)
        return X, y

    def train(self, train_df):
        train_df['user_idx'] = self.user_encoder.transform(train_df[self.user_clm])
        train_df['item_idx'] = self.item_encoder.transform(train_df[self.item_clm])
        self.train_df = train_df
        X, y = self.build_sequences(train_df)
        dataset = SequenceDataset(X, y, self.num_items, allowed_indices=self.allowed_item_indices, max_len=self.args.MAX_SEQ_LEN)
        loader = DataLoader(dataset, batch_size=self.args.BATCH_SIZE, shuffle=True)

        self.model = TransformerStatusEncoder(
            vocab_size=self.num_items,
            embed_dim=self.args.EMBED_DIM,
            num_heads=self.args.NUM_HEADS,
            num_layers=self.args.NUM_LAYERS,
            dropout=self.args.DROPOUT,
            max_len=self.args.MAX_SEQ_LEN
        ).to(self.device)

        self.load_llm_embeddings(freeze=True)
        self.model.set_embedding_bank(self.embedding_bank, use_avg=False, n_desc=self.n_desc)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.LR)
        self.model.train()


        for epoch in range(self.args.NUM_EPOCHS):
            current_temp = cosine_anneal_temperature(epoch, self.args.NUM_EPOCHS) #
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                status_emb = self.model(batch_x)

                
                item_embs = self.embedding_bank.mean(dim=1)
                loss = multi_positive_infonce_loss(status_emb, batch_y, item_embs, self.allowed_item_indices, temperature=current_temp)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # if (epoch + 1) % self.args.EVAL_TRAIN_EVERY == 0:
            #     self.evaluate_on_train_split(k=5, top_k=5)

            print(f"Epoch {epoch+1}/{self.args.NUM_EPOCHS} - Loss: {total_loss/len(loader):.4f}")

    def evaluate_on_train_split(self, k=5, top_k=5):
        user_seqs = self.train_df.groupby('user_idx')['item_idx'].apply(list)
        recall_total, precision_total, hit_total, mrr_total, total_users = 0, 0, 0, 0, 0

        self.model.eval()
        self.model.set_embedding_bank(self.embedding_bank, use_avg=True, n_desc=self.n_desc)

        with torch.no_grad():
            cnt = 0
            for user_idx, seq in user_seqs.items():
                if len(seq) <= k:
                    continue
                cnt += 1
                prefix = seq[:k]
                future = set(seq[k:])

                if not future:
                    continue


                input_tensor = pad_and_truncate(prefix, self.args.MAX_SEQ_LEN).unsqueeze(0).to(self.device)
                status_emb = self.model(input_tensor)
                item_embs = self.embedding_bank.mean(dim=1)

                scores = F.cosine_similarity(status_emb, item_embs.unsqueeze(0), dim=-1).squeeze()
                topk_indices = torch.topk(scores, top_k).indices.cpu().tolist()
                recommended_items = [self.idx2item[i] for i in topk_indices]
                future = set([self.idx2item[i] for i in future])

                hits_k = set(recommended_items) & future
                recall_total += len(hits_k) / len(future)
                precision_total += len(hits_k) / top_k
                hit_total += 1 if hits_k else 0
                for rank, item in enumerate(recommended_items, start=1):
                    if item in future:
                        mrr_total += 1 / rank
                        break
                total_users += 1

        recall = recall_total / total_users if total_users > 0 else 0
        precision = precision_total / total_users if total_users > 0 else 0
        hit_rate = hit_total / total_users if total_users > 0 else 0
        mrr = mrr_total / total_users if total_users > 0 else 0

        print(f"[Train-Eval k={k}, {cnt} users] 🎯 Recall@{top_k}: {recall:.4f} | 📐 Precision@{top_k}: {precision:.4f} | ✅ Hit Rate: {hit_rate:.4f} | 📈 MRR: {mrr:.4f}")


    def evaluate(self, val_df, top_k=5, save_log=False):
        train_histories = self.train_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
        val_df = val_df[val_df[self.user_clm].isin(self.user2idx) & val_df[self.item_clm].isin(self.item_encoder.classes_)].copy()
        val_df['user_idx'] = self.user_encoder.transform(val_df[self.user_clm])
        val_df['item_idx'] = self.item_encoder.transform(val_df[self.item_clm])

        recall_total, precision_total, hit_total, mrr_total, total_users = 0, 0, 0, 0, 0
        self.model.eval()
        self.model.set_embedding_bank(self.embedding_bank, use_avg=True, n_desc=self.n_desc)

        if self.args.USE_VAL_DISC:
            val_conf_disc = self.args.VAL_DISCOUNT_RATIO
        else:
            val_conf_disc = 1.0

        predict_dict = dict()

        with torch.no_grad():
            for user_idx, group in val_df.groupby('user_idx'):
                history = train_histories.get(user_idx, [])
                if not history:
                    continue
                if len(history) > self.args.MAX_SEQ_LEN:
                    history = history[-self.args.MAX_SEQ_LEN:]

                input_tensor = pad_and_truncate(history, self.args.MAX_SEQ_LEN).unsqueeze(0).to(self.device)
                status_emb = self.model(input_tensor)
                item_embs = self.embedding_bank.mean(dim=1)

                scores = F.cosine_similarity(status_emb, item_embs.unsqueeze(0), dim=-1).squeeze()
                scores[self.validation_idx] *= val_conf_disc
                
                
                topk_indices = torch.topk(scores, top_k).indices.cpu().tolist()
                recommended_items = [self.idx2item[i] for i in topk_indices]
                true_items = set(group[self.item_clm])
                hits_k = set(recommended_items) & true_items

                if top_k == 5:
                    predict_dict[user_idx] = {"true": list(true_items), "predict": recommended_items}

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


        if save_log:
            os.makedirs(self.SAVE_PATH, exist_ok=True)
            with open(self.SAVE_PATH + f"Transformer_Recommender_top{top_k}.txt", "w") as f:
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

