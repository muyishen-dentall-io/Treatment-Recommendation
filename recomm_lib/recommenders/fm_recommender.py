from .base import Recommender
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from fastFM.bpr import FMRecommender
from scipy.sparse import csr_matrix
import pdb
from tqdm import tqdm
import os
from collections import defaultdict
import random


class FM_Recommender(Recommender):

    def __init__(self, args):
        self.args = args
        self.model = FMRecommender(
            n_iter=args.NUM_ITER,
            rank=args.NUM_FACTORS,
            l2_reg=args.REGULARIZATION,
            random_state=args.SEED,
            step_size=0.1
        )

        self.SAVE_PATH = 'results/FM_Recommender/'

        self.encoder = OneHotEncoder(handle_unknown="ignore")

    def _compute_age_group(self, df):
        df["date"] = pd.to_datetime(df["date"])
        df["birthday"] = pd.to_datetime(df["birthday"])
        df["age_at_treatment"] = ((df["date"] - df["birthday"]).dt.days / 365.25).astype(int)
        # df["age_group"] = pd.cut(df["age_at_treatment"], bins=[0, 18, 25, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], labels=False)
        df["age_group"] = pd.cut(df["age_at_treatment"], bins=[0, 18, 35, 50, 65, 90, 120], labels=False)
        return df

    def train(self, train_df):
        self.train_df = train_df
        merged = self.train_df

        merged = self._compute_age_group(merged)
        merged["age_group"] = merged["age_group"].fillna(-1).astype(int)

        # features = merged[["user_idx", "item_idx", "smoking", "gender", "age_group"]].fillna("Unknown")
        features = merged[["user_idx", "item_idx", "gender"]].fillna("Unknown")
        # features = merged[["user_idx", "item_idx", "age_group"]].fillna("Unknown")
        self.encoder.fit(features)
        self.X_train = self.encoder.transform(features)

        os.makedirs("cache", exist_ok=True)
        pair_indices_path = "cache/fm_pair_indices.npy"

        if os.path.exists(pair_indices_path) and os.path.exists("cache/fm_merged_with_synthetic_rows.parquet"):
            print("✅ Loading cached pair_indices and merged...")
            self.pair_indices = np.load(pair_indices_path)
            merged = pd.read_parquet("cache/fm_merged_with_synthetic_rows.parquet")
        else:
            print("🔄 Generating pair_indices...")
            user_pos_items = defaultdict(set)
            for row in merged.itertuples():
                user_pos_items[row.user_idx].add(row.item_idx)

            all_items = set(merged["item_idx"].unique())

            pair_indices = []
            for user_idx, pos_items in tqdm(user_pos_items.items()):
                pos_items = list(pos_items)
                for pos_item in pos_items:
                    for _ in range(1):  # number of negative samples per positive
                        neg_item = random.choice(list(all_items - set(pos_items)))

                        pos_row = merged[(merged.user_idx == user_idx) & (merged.item_idx == pos_item)]
                        neg_row = merged[(merged.user_idx == user_idx) & (merged.item_idx == neg_item)]

                        if pos_row.empty:
                            continue
                        if neg_row.empty:
                            neg_row = pos_row.copy()
                            neg_row = neg_row.iloc[0:1].copy()
                            neg_row.loc[:, "item_idx"] = neg_item
                            merged = pd.concat([merged, neg_row], ignore_index=True)
                            neg_row_index = merged.index[-1]
                        else:
                            neg_row_index = neg_row.index[0]

                        pair_indices.append((pos_row.index[0], neg_row_index))

            self.pair_indices = np.array(pair_indices)
            np.save(pair_indices_path, self.pair_indices)
            merged.to_parquet("cache/fm_merged_with_synthetic_rows.parquet")


        # self.X_train = self.encoder.transform(merged[["user_idx", "item_idx", "smoking", "gender", "age_group"]].fillna("Unknown"))
        self.X_train = self.encoder.transform(merged[["user_idx", "item_idx", "gender"]].fillna("Unknown"))
        # self.X_train = self.encoder.transform(merged[["user_idx", "item_idx", "age_group"]].fillna("Unknown"))
        
        # pdb.set_trace()

        self.model.fit(self.X_train, self.pair_indices)

        # pdb.set_trace()
        return

    def evaluate(self, val_df, top_k):
        val_df = val_df[val_df["user_id"].isin(self.user2idx) & val_df["item"].isin(self.item2idx)].copy()
        self.patient_info_df = pd.read_csv("data/patient_info.csv")
        val_df.loc[:, "user_idx"] = val_df["user_id"].map(self.user2idx)
        val_df.loc[:, "item_idx"] = val_df["item"].map(self.item2idx)

        all_items = list(self.item2idx.values())
        recall_total = precision_total = hit_total = mrr_total = total_users = 0

        for user_id in tqdm(val_df[self.user_clm].unique()):
            if user_id not in self.user2idx:
                continue

            user_idx = self.user2idx[user_id]
            true_items = set(val_df[val_df[self.user_clm] == user_id][self.item_clm])

            if not true_items:
                continue

            user_info = self.patient_info_df[self.patient_info_df["patient_id"] == user_id]
            if user_info.empty:
                continue

            # Get the user's latest training date and use that to simulate the prediction point (age + 1)
            train_dates = self.train_df[self.train_df["user_id"] == user_id]["date"]
            if train_dates.empty:
                continue
            last_train_date = pd.to_datetime(train_dates.max())
            birthday = pd.to_datetime(user_info["birthday"].values[0])
            age = int((last_train_date - birthday).days / 365.25)
            # age_group = pd.cut([age], bins=[0, 18, 25, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], labels=False)[0]
            age_group = pd.cut([age], bins=[0, 18, 35, 50, 65, 90, 120], labels=False)[0]
            

            candidate_rows = []
            for item_id in all_items:
                if item_id not in self.idx2item:
                    continue

                row = {
                    "user_idx": user_idx,
                    "item_idx": item_id,
                    # "smoking": str(user_info["smoking"].values[0]),
                    "gender": str(user_info["gender"].values[0]),
                    # "age_group": int(age_group) if not pd.isna(age_group) else -1
                }
                candidate_rows.append(row)

            candidate_df = pd.DataFrame(candidate_rows).fillna("Unknown")


            scores = self.model.predict(self.encoder.transform(candidate_df))

            # pdb.set_trace()

            candidate_df["score"] = scores
            top_items = candidate_df.sort_values("score", ascending=False)["item_idx"].tolist()[:top_k]
            top_items = [self.idx2item[i] for i in top_items]

            hits = set(top_items) & true_items

            # pdb.set_trace()

            recall_total += len(hits) / len(true_items)
            precision_total += len(hits) / top_k
            hit_total += 1 if hits else 0
            mrr_total += next((1 / (i + 1) for i, item in enumerate(top_items) if item in true_items), 0)
            total_users += 1

        recall = recall_total / total_users if total_users > 0 else 0
        precision = precision_total / total_users if total_users > 0 else 0
        hit_rate = hit_total / total_users if total_users > 0 else 0
        mrr = mrr_total / total_users if total_users > 0 else 0

        os.makedirs(self.SAVE_PATH, exist_ok=True)

        with open(self.SAVE_PATH + "FM_Recommender_top{}.txt".format(top_k), "w", encoding="utf-8") as f:
            def log(msg):
                print(msg)
                f.write(msg + "\n")

            log(f"🎯 Recall@{top_k}:         {recall:.4f}")
            log(f"📐 Precision@{top_k}:      {precision:.4f}")
            log(f"✅ Hit Rate@{top_k}:       {hit_rate:.4f}")
            log(f"📈 MRR@{top_k}:            {mrr:.4f}")