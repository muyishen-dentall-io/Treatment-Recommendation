import recomm_lib
import pdb
from types import SimpleNamespace
import pandas as pd

# DB_PATH = "patients.db"

# df = recomm_lib.load_treatment_data(DB_PATH)

# df = recomm_lib.filter_data(df)

# # train_df, val_df = recomm_lib.train_test_split_by_ratio(df, ratio=0.2)
# train_df, val_df = recomm_lib.train_test_split_by_n_year(df)

# train_df.to_csv('data/train_df.csv')
# val_df.to_csv('data/val_df.csv')

train_df = pd.read_csv('data/train_df.csv')
val_df = pd.read_csv('data/val_df.csv')

# args = SimpleNamespace(SEED=42, NUM_FACTORS = 32, NUM_ITER = 20, REGULARIZATION = 0.1, ALPHA = 50)

# recommender = recomm_lib.CF_Recommender(args)
# recommender = recomm_lib.LLM_Recommender(mode='note')
# recommender = recomm_lib.LLM_Recommender(mode='treatment')
recommender = recomm_lib.LLM_Recommender(mode='all')

train_df = recommender.user_item2idx(train_df, "user_id", "item")

recommender.train(train_df)

recommender.evaluate(val_df, top_k=5)

