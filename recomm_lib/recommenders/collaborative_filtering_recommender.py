from .base import Recommender
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
import os

class CF_Recommender(Recommender):
    
    def __init__(self, args):
        
        self.args = args
        self.SAVE_PATH = 'results/CF_Recommender/'

        return
    
    def train(self, train_df):
        self.item_user_matrix = coo_matrix(
            (np.ones(len(train_df)), (train_df["item_idx"], train_df["user_idx"]))
        )
        
        self.model = AlternatingLeastSquares(
            factors=self.args.NUM_FACTORS,
            regularization=self.args.REGULARIZATION,
            iterations=self.args.NUM_ITER,
            random_state=self.args.SEED
        )
        self.model.fit((self.item_user_matrix * self.args.ALPHA).T)

        return
    
    def predict(self):
        
        return
    
    def evaluate(self, val_df, top_k):
        original_val_len = len(val_df)
        
        val_df = val_df[val_df["user_id"].isin(self.user2idx) & val_df["item"].isin(self.item2idx)]
        filtered_val_len = len(val_df)
        dropped_val_len = original_val_len - filtered_val_len

        print(f"✅ Validation interactions (after filtering): {filtered_val_len}")
        print(f"⚠️ Dropped validation interactions: {dropped_val_len} ({dropped_val_len/original_val_len:.1%})")

        recall_total = 0
        precision_total = 0
        hit_total = 0
        mrr_total = 0
        total_users = 0
        
        for user_id, group in val_df.groupby(self.user_clm):
            if user_id not in self.user2idx:
                continue

            user_idx = self.user2idx[user_id]
            true_items = set(group[self.item_clm])
            true_count = len(true_items)

            if true_count == 0:
                continue
            
            recommended_k, _ = self.model.recommend(
                user_idx, self.item_user_matrix.T, N=top_k, filter_already_liked_items=False
            )
            recommended_k = [self.idx2item[i] for i in recommended_k]
  
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
        
        os.makedirs(self.SAVE_PATH, exist_ok=True)
        
        with open(self.SAVE_PATH + "CF_Recommender_top{}.txt".format(top_k), "w", encoding="utf-8") as f:
            def log(msg):
                print(msg)
                f.write(msg + "\n")

            log(f"🎯 Recall@{top_k}:         {recall:.4f}")
            log(f"📐 Precision@{top_k}:      {precision:.4f}")
            log(f"✅ Hit Rate@{top_k}:       {hit_rate:.4f}")
            log(f"📈 MRR@{top_k}:            {mrr:.4f}")
    