from .base import Recommender
import pdb
from tqdm import tqdm
import os

class Static_Recommender(Recommender):
    def __init__(self):

        self.predict_order = [
            '91004C牙結石清除－全口', '91014C牙周暨齲齒控制基本處置',
             '90012C橡皮障防濕裝置', '89009C後牙複合樹脂充填-雙面', '01271C環口全景X光初診診察',
             '89010C後牙複合樹脂充填-三面', '91020C牙菌斑去除照護', 
             '90015C根管開擴及清創',
             '91001C牙周病緊急處置', '81氟化防齲處理(包括牙醫師專業塗氟處理、一般性口腔檢查、衛生教育）', 
             '92001C非特定局部治療', '34002C咬翼式 X光攝影', '92014C複雜性拔牙', '34001C根尖周 X光攝影',
             '89008C後牙複合樹脂充填-單面', 'RCT療程治療中繼',
             '00315C符合牙醫門診加強感染管制實施方案之環口全景X光初診診察', '89012C前牙三面複合樹脂充填',
             '89005C前牙複合樹脂充填-雙面', '89008C後牙複合樹脂充填-單面'
        ]

        self.SAVE_PATH = 'results/Static_Recommender/'
        return
    
    def predict(self, top_k):
        
        return self.predict_order[:top_k]
    
    def train(self, train_df):
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

        for user_id, group in tqdm(val_df.groupby(self.user_clm)):
            if user_id not in self.user2idx:
                continue

            user_idx = self.user2idx[user_id]
            true_items = set(group[self.item_clm])
            true_count = len(true_items)

            if true_count == 0:
                continue

            recommended_k = self.predict(top_k)

            # pdb.set_trace()

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

        with open(self.SAVE_PATH + "Static_Recommender_top{}.txt".format(top_k), "w", encoding="utf-8") as f:
            def log(msg):
                print(msg)
                f.write(msg + "\n")

            log(f"🎯 Recall@{top_k}:         {recall:.4f}")
            log(f"📐 Precision@{top_k}:      {precision:.4f}")
            log(f"✅ Hit Rate@{top_k}:       {hit_rate:.4f}")
            log(f"📈 MRR@{top_k}:            {mrr:.4f}")

        return