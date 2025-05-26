import pdb
from tqdm import tqdm
import os


class Static_Recommender:
    def __init__(self, args):
        
        self.args = args
        self.SAVE_PATH = "results/Static_Recommender/"
        return

    def user_item2idx(self, df, user_clm="user_id", item_clm="item"):
        # user_clm = "user_id", item_clm = "item_id"
        self.user_clm = user_clm
        self.item_clm = item_clm
        user_ids = df[user_clm].unique()
        item_ids = df[item_clm].unique()
        self.user2idx = {uid: i for i, uid in enumerate(user_ids)}
        self.item2idx = {iid: i for i, iid in enumerate(item_ids)}
        self.idx2item = {i: iid for iid, i in self.item2idx.items()}
        self.idx2user = {i: iid for iid, i in self.user2idx.items()}

        df["user_idx"] = df[user_clm].map(self.user2idx)
        df["item_idx"] = df[item_clm].map(self.item2idx)

        return df

    def predict(self, top_k):

        return self.predict_order[:top_k]

    def train(self, train_df):

        self.predict_order = train_df["item"].value_counts().head(10).index.tolist()

        return

    def evaluate(self, val_df, top_k, save_log=False):

        original_val_len = len(val_df)

        val_df = val_df[
            val_df["user_id"].isin(self.user2idx) & val_df["item"].isin(self.item2idx)
        ]
        filtered_val_len = len(val_df)
        dropped_val_len = original_val_len - filtered_val_len

        print(f"Validation interactions (after filtering): {filtered_val_len}")
        print(
            f"Dropped validation interactions: {dropped_val_len} ({dropped_val_len/original_val_len:.1%})"
        )

        recall_total = 0
        precision_total = 0
        hit_total = 0
        mrr_total = 0
        total_users = 0

        for user_id, group in tqdm(val_df.groupby(self.user_clm)):
            if user_id not in self.user2idx:
                continue

            user_idx = self.user2idx[user_id]
            true_items = set(group[self.item_clm])
            true_count = len(true_items)

            if true_count == 0:
                continue

            recommended_k = self.predict(top_k)

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

            with open(
                self.SAVE_PATH + "Static_Recommender_top{}.txt".format(top_k),
                "w",
                encoding="utf-8",
            ) as f:

                def log(msg):
                    print(msg)
                    f.write(msg + "\n")

                log(f"🎯 Recall@{top_k}:         {recall:.4f}")
                log(f"📐 Precision@{top_k}:      {precision:.4f}")
                log(f"✅ Hit Rate@{top_k}:       {hit_rate:.4f}")
                log(f"📈 MRR@{top_k}:            {mrr:.4f}")

            return

        else:

            print(f"🎯 Recall@{top_k}:         {recall:.4f}")
            print(f"📐 Precision@{top_k}:      {precision:.4f}")
            print(f"✅ Hit Rate@{top_k}:       {hit_rate:.4f}")
            print(f"📈 MRR@{top_k}:            {mrr:.4f}")
