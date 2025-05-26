import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import json
import os
import pdb


class NCFDataset(Dataset):
    def __init__(self, df):
        self.user = torch.tensor(df['user_idx'].values, dtype=torch.long)
        self.item = torch.tensor(df['item_idx'].values, dtype=torch.long)
        self.gender = torch.tensor(df['gender_idx'].values, dtype=torch.long)
        self.label = torch.tensor(df['label'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.user)

    def __getitem__(self, idx):
        return self.user[idx], self.item[idx], self.gender[idx], self.label[idx]


class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, num_genders, embed_dim=64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.gender_embedding = nn.Embedding(num_genders, embed_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + embed_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_idx, item_idx, gender_idx):
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        gender_embed = self.gender_embedding(gender_idx)
        x = torch.cat([user_embed, item_embed, gender_embed], dim=1)
        return self.mlp(x).squeeze()


class NCF_Recommender:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SAVE_PATH = 'results/NCF_Recommender/'

    def user_item2idx(self, df, user_col='user_id', item_col='item', gender_col='gender'):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()

        df['user_idx'] = self.user_encoder.fit_transform(df[user_col])
        df['item_idx'] = self.item_encoder.fit_transform(df[item_col])
        df['gender_idx'] = self.gender_encoder.fit_transform(df[gender_col])

        self.num_users = df['user_idx'].nunique()
        self.num_items = df['item_idx'].nunique()
        self.num_genders = df['gender_idx'].nunique()
        self.user_clm = user_col
        self.item_clm = item_col
        self.user2idx = dict(zip(self.user_encoder.classes_, range(len(self.user_encoder.classes_))))
        self.idx2item = dict(enumerate(self.item_encoder.classes_))
        return df

    def train(self, train_df):

        pos_df = train_df.copy()
        pos_df['label'] = 1

        neg_samples = []
        user_item_set = set(zip(pos_df['user_idx'], pos_df['item_idx']))
        for u in pos_df['user_idx'].unique():
            gender_idx = pos_df[pos_df['user_idx'] == u]['gender_idx'].iloc[0]
            for _ in range(self.args.NEG_PER_USER):
                while True:
                    j = np.random.randint(self.num_items)
                    if (u, j) not in user_item_set:
                        neg_samples.append([u, j, gender_idx, 0])
                        break

        neg_df = pd.DataFrame(neg_samples, columns=['user_idx', 'item_idx', 'gender_idx', 'label'])
        full_df = pd.concat([pos_df[['user_idx', 'item_idx', 'gender_idx', 'label']], neg_df])

        dataset = NCFDataset(full_df)
        loader = DataLoader(dataset, batch_size=self.args.BATCH_SIZE, shuffle=True)

        self.model = NeuralCollaborativeFiltering(
            self.num_users, self.num_items, self.num_genders, embed_dim=self.args.EMBED_DIM
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.LR)
        loss_fn = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.args.NUM_EPOCHS):
            total_loss = 0
            for user, item, gender, label in loader:
                user, item, gender, label = user.to(self.device), item.to(self.device), gender.to(self.device), label.to(self.device)
                pred = self.model(user, item, gender)
                loss = loss_fn(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.args.NUM_EPOCHS} - Loss: {total_loss:.4f}")

    def recommend(self, user_id, top_k=5):
        self.model.eval()
        user_idx = self.user_encoder.transform([user_id])[0]
        gender_idx = self.gender_encoder.transform(["MALE"])[0]
        all_items = torch.arange(self.num_items).to(self.device)
        user_tensor = torch.tensor([user_idx] * self.num_items).to(self.device)
        gender_tensor = torch.tensor([gender_idx] * self.num_items).to(self.device)
        with torch.no_grad():
            scores = self.model(user_tensor, all_items, gender_tensor)
        top_items = torch.topk(scores, top_k).indices.cpu().numpy()
        return self.item_encoder.inverse_transform(top_items)

    def evaluate(self, val_df, top_k, save_log=False):
        original_val_len = len(val_df)
        val_df = val_df[val_df[self.user_clm].isin(self.user2idx) & val_df[self.item_clm].isin(self.item_encoder.classes_)].copy()
        filtered_val_len = len(val_df)
        dropped_val_len = original_val_len - filtered_val_len

        print(f"Validation interactions (after filtering): {filtered_val_len}")
        print(f"Dropped validation interactions: {dropped_val_len/original_val_len:.1%}")

        recall_total = 0
        precision_total = 0
        hit_total = 0
        mrr_total = 0
        total_users = 0

        val_df['user_idx'] = self.user_encoder.transform(val_df[self.user_clm].values)
        val_df['item_idx'] = self.item_encoder.transform(val_df[self.item_clm].values)
        val_df['gender_idx'] = self.gender_encoder.transform(val_df['gender'].values)

        for user_id, group in val_df.groupby(self.user_clm):
            if user_id not in self.user2idx:
                continue

            user_idx = self.user2idx[user_id]
            gender_idx = group['gender_idx'].iloc[0]
            true_items = set(group[self.item_clm])
            true_count = len(true_items)
            if true_count == 0:
                continue

            self.model.eval()
            all_items = torch.arange(self.num_items).to(self.device)
            user_tensor = torch.tensor([user_idx] * self.num_items).to(self.device)
            gender_tensor = torch.tensor([gender_idx] * self.num_items).to(self.device)
            with torch.no_grad():
                scores = self.model(user_tensor, all_items, gender_tensor)
            top_k_items = torch.topk(scores, top_k).indices.cpu().numpy()
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

            with open(self.SAVE_PATH + "NCF_Recommender_top{}.txt".format(top_k), "w", encoding="utf-8") as f:
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


class NCF_Recommender_GNNemb(NCF_Recommender):
    def __init__(self, args):
        super().__init__(args)
        self.SAVE_PATH = "results/NCF_GNN_Recommender/"
    

    def load_gnn_embeddings(self, path="./emb_generate/gnn/treatment_embeddings.json", freeze=True):
        with open(path, "r") as f:
            loaded_embeddings = json.load(f)

        with torch.no_grad():
            for item, emb in loaded_embeddings.items():
                try:
                    idx = int(self.item_encoder.transform(np.array([item])))
                    emb_tensor = torch.tensor(emb, dtype=torch.float, device=self.device)
                    self.model.item_embedding.weight[idx] = emb_tensor
                except Exception as e:
                    print(f"Warning: Could not load embedding for item {item}: {e}")

        if freeze:
            self.model.item_embedding.weight.requires_grad = False
            print(f"✅ Loaded {len(loaded_embeddings)} embeddings and **froze** item embedding.")
        else:
            print(f"✅ Loaded {len(loaded_embeddings)} embeddings (item embedding remains trainable).")
        
        return


    def train(self, train_df):
        pos_df = train_df.copy()
        pos_df['label'] = 1

        neg_samples = []
        user_item_set = set(zip(pos_df['user_idx'], pos_df['item_idx']))
        for u in pos_df['user_idx'].unique():
            gender_idx = pos_df[pos_df['user_idx'] == u]['gender_idx'].iloc[0]
            for _ in range(self.args.NEG_PER_USER):
                while True:
                    j = np.random.randint(self.num_items)
                    if (u, j) not in user_item_set:
                        neg_samples.append([u, j, gender_idx, 0])
                        break

        neg_df = pd.DataFrame(neg_samples, columns=['user_idx', 'item_idx', 'gender_idx', 'label'])
        full_df = pd.concat([pos_df[['user_idx', 'item_idx', 'gender_idx', 'label']], neg_df])

        dataset = NCFDataset(full_df)
        loader = DataLoader(dataset, batch_size=self.args.BATCH_SIZE, shuffle=True)

        self.model = NeuralCollaborativeFiltering(
            self.num_users, self.num_items, self.num_genders, embed_dim=self.args.EMBED_DIM
        ).to(self.device)

        self.load_gnn_embeddings()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.LR)
        loss_fn = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.args.NUM_EPOCHS):
            total_loss = 0
            for user, item, gender, label in loader:
                user, item, gender, label = user.to(self.device), item.to(self.device), gender.to(self.device), label.to(self.device)
                pred = self.model(user, item, gender)
                loss = loss_fn(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.args.NUM_EPOCHS} - Loss: {total_loss:.4f}")

    def evaluate(self, val_df, top_k, save_log=False):
        original_val_len = len(val_df)
        val_df = val_df[val_df[self.user_clm].isin(self.user2idx) & val_df[self.item_clm].isin(self.item_encoder.classes_)].copy()
        filtered_val_len = len(val_df)
        dropped_val_len = original_val_len - filtered_val_len

        print(f"✅ Validation interactions (after filtering): {filtered_val_len}")
        print(f"⚠️ Dropped validation interactions: {dropped_val_len/original_val_len:.1%}")

        recall_total = 0
        precision_total = 0
        hit_total = 0
        mrr_total = 0
        total_users = 0

        val_df['user_idx'] = self.user_encoder.transform(val_df[self.user_clm].values)
        val_df['item_idx'] = self.item_encoder.transform(val_df[self.item_clm].values)
        val_df['gender_idx'] = self.gender_encoder.transform(val_df['gender'].values)

        for user_id, group in val_df.groupby(self.user_clm):
            if user_id not in self.user2idx:
                continue

            user_idx = self.user2idx[user_id]
            gender_idx = group['gender_idx'].iloc[0]
            true_items = set(group[self.item_clm])
            true_count = len(true_items)
            if true_count == 0:
                continue

            self.model.eval()
            all_items = torch.arange(self.num_items).to(self.device)
            user_tensor = torch.tensor([user_idx] * self.num_items).to(self.device)
            gender_tensor = torch.tensor([gender_idx] * self.num_items).to(self.device)
            with torch.no_grad():
                scores = self.model(user_tensor, all_items, gender_tensor)
            top_k_items = torch.topk(scores, top_k).indices.cpu().numpy()
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

            with open(self.SAVE_PATH + "NCF_GNN_Recommender_top{}.txt".format(top_k), "w", encoding="utf-8") as f:
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