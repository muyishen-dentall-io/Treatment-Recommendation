from .base import Recommender
from openai import OpenAI
from dotenv import load_dotenv , find_dotenv
import os
import re
from tqdm import tqdm
import pdb
import json
import sqlite3
import pandas as pd

class LLM_Recommender(Recommender):
    def __init__(self, mode='treatment', llm_model="gpt-4.1"):
        
        _ = load_dotenv(find_dotenv())
        
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.mode = mode
        self.llm_model = llm_model

        self.client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
        return
    
    def train(self, train_df):
        self.train_df = train_df

    def predict(self, llm_prompt):
        
        system_prompt = "Please act like a dentist and predict what a patient will but based on his past information"

        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": llm_prompt,
                },
            ],
            temperature=0.0
        )

        self.prompt_tokens += completion.usage.prompt_tokens
        self.completion_tokens += completion.usage.completion_tokens

        reply = completion.choices[0].message.content
        
        return reply
    
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

        predict_dict = dict()

        for user_id, group in tqdm(val_df.groupby(self.user_clm)):

            if user_id not in self.user2idx:
                continue

            user_idx = self.user2idx[user_id]
            true_items = set(group[self.item_clm])
            true_count = len(true_items)

            if true_count == 0:
                continue


            history_list = self.train_df[self.train_df["user_idx"]==user_idx]["item"].to_list()
            llm_prompt = self.history2prompt(user_idx, history_list, top_k)

            # pdb.set_trace()
            reply = self.predict(llm_prompt)
            
            recommended_k = self.reply2recomm_list(reply)

            predict_dict[user_id] = {"true": list(true_items), "predict": recommended_k}
            
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
        
        with open("results/LLM_Recommender-{}.txt".format(self.mode), "w", encoding="utf-8") as f:
            def log(msg):
                print(msg)
                f.write(msg + "\n")

            log(f"🎯 Recall@{top_k}:         {recall:.4f}")
            log(f"📐 Precision@{top_k}:      {precision:.4f}")
            log(f"✅ Hit Rate@{top_k}:       {hit_rate:.4f}")
            log(f"📈 MRR@{top_k}:            {mrr:.4f}")
            log("🧮 Average prompt_tokens: {:.2f}".format(self.prompt_tokens / total_users))
            log("🧾 Average completion_tokens: {:.2f}".format(self.completion_tokens / total_users))
            log("👥 Total Users: {}".format(total_users))

        with open("results/{}_{}_predict.json".format(self.llm_model, self.mode), "w", encoding="utf-8") as f:
            json.dump(predict_dict, f, ensure_ascii=False, indent=2)
    
    
    def history2prompt(self, user_idx, history_list, top_k):
        with open("data/predict_prompt.txt", "r", encoding="utf-8") as f:
                query_prompt = f.read()
        
        prompt_end = "，請根據這些資訊輸出你覺得病患最有可能做的{}個處置，只要以(數字開頭)條列式回覆這些處置名稱即可".format(top_k, top_k)

        if self.mode == 'treatment':
            query_prompt += "病人曾經做過的處置包含這些: "
            query_prompt += '， '.join(history_list)
            query_prompt += prompt_end

        elif self.mode == 'note':
            user_id = self.idx2user[user_idx]
            conn = sqlite3.connect("patients.db")
            df = pd.read_sql_query(
                """
                SELECT doctor_comment
                FROM dental_treatments
                WHERE patient_id = ?
                """,
                conn,
                params=(int(user_id),)
            )
            conn.close()

            comments = '\n'.join(df["doctor_comment"].fillna('').tolist())

            query_prompt += "，過去的牙醫曾經為病人做過以下紀錄: "
            query_prompt += comments
            query_prompt += prompt_end
        
        elif self.mode == 'all':
            user_id = self.idx2user[user_idx]
            conn = sqlite3.connect("patients.db")
            df = pd.read_sql_query(
                """
                SELECT doctor_comment
                FROM dental_treatments
                WHERE patient_id = ?
                """,
                conn,
                params=(int(user_id),)
            )
            conn.close()

            comments = '\n'.join(df["doctor_comment"].fillna('').tolist())
            
            query_prompt += "病人曾經做過的處置包含這些: "
            query_prompt += '， '.join(history_list)
            query_prompt += "，過去的牙醫曾經為病人做過以下紀錄: "
            query_prompt += comments
            query_prompt += prompt_end

        return query_prompt
    
    def reply2recomm_list(self, reply):
        return [s.strip() for s in re.findall(r"\d+\.\s+([^\n]+)", reply)]