import recomm_lib
import pdb
from types import SimpleNamespace
import pandas as pd
import random
import numpy as np
import torch
from recomm_lib import (
    CF_Recommender,
    LLM_Recommender,
    Random_Recommender,
    Static_Recommender,
    BPR_Recommender,
    FM_Recommender,
    NCF_Recommender,
    NCF_Recommender_GNNemb,
    Next_Prediction_Recommender,
    TransformerRecommender
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # if using GPU
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------
# Split data into training and validation

# DB_PATH = "patients.db"

# df = recomm_lib.load_treatment_data(DB_PATH)

# df = recomm_lib.filter_data(df)

# # train_df, val_df = recomm_lib.train_test_split_by_ratio(df, ratio=0.2)
# train_df, val_df = recomm_lib.train_test_split_by_n_year(df)

# train_df.to_csv('data/train_df.csv')
# val_df.to_csv('data/val_df.csv')

# -----------------------------------------------

train_df = pd.read_csv('data/train_df.csv')
# train_df = pd.read_csv('data/new_train_df.csv')
val_df = pd.read_csv('data/val_df.csv')

# train_df = pd.read_csv('data/train_df_w_info.csv')
# val_df = pd.read_csv('data/val_df_w_info.csv')

recommender_configs = [
    # {
    #     "name": "CF_Recommender",
    #     "recommender": CF_Recommender(
    #         SimpleNamespace(SEED=42, NUM_FACTORS=32, NUM_ITER=100, REGULARIZATION=0.1, ALPHA=20)
    #     )
    # },
    # {"name": "LLM_Recommender_note", "recommender": LLM_Recommender(mode="note")},
    # {"name": "LLM_Recommender_treatment", "recommender": LLM_Recommender(mode="treatment")},
    # {"name": "LLM_Recommender_all", "recommender": LLM_Recommender(mode="all")},
    # {"name": "Random_Recommender", "recommender": Random_Recommender()},
    # {"name": "Static_Recommender", "recommender": Static_Recommender()},
    # {
    #     "name": "BPR_Recommender",
    #     "recommender": BPR_Recommender(
    #         SimpleNamespace(SEED=42, NUM_FACTORS=64, NUM_ITER=50, REGULARIZATION=0.01)
    #     ),
    # },
    # {
    #     "name": "FM_Recommender",
    #     "recommender": FM_Recommender(
    #         SimpleNamespace(SEED=42, NUM_FACTORS=32, NUM_ITER=400, REGULARIZATION=0.01)
    #     ),
    # },
    # {
    #     "name": "NCF_Recommender",
    #     "recommender": NCF_Recommender(
    #         SimpleNamespace(NUM_EPOCHS=5, NEG_PER_USER=20, EMBED_DIM=128, BATCH_SIZE=128, LR=1e-3, SEED=42)
    #     ),
    # },
    # {
    #     "name": "NCF_Recommender_GNNemb",
    #     "recommender": NCF_Recommender_GNNemb(
    #         SimpleNamespace(NUM_EPOCHS=8, NEG_PER_USER=20, EMBED_DIM=128, BATCH_SIZE=128, LR=1e-3, SEED=42)
    #     ),
    # },
    # {
    #     "name": "Next_Prediction_Recommender",
    #     "recommender": Next_Prediction_Recommender(
    #         SimpleNamespace(
    #             BATCH_SIZE=128,
    #             EMBED_DIM=128,
    #             NUM_HEADS=2,
    #             NUM_LAYERS=1,
    #             DROPOUT=0.1,
    #             LR=1e-4,
    #             NUM_EPOCHS=6,
    #             MAX_SEQ_LEN=8,
    #             DECAY_ALPHA=1.0
    #         )
    #     ),
    # },
    # {
    #     "name": "Next_Prediction_Recommender",
    #     "recommender": Next_Prediction_Recommender(
    #         SimpleNamespace(
    #             BATCH_SIZE=128,
    #             EMBED_DIM=256,
    #             NUM_HEADS=2,
    #             NUM_LAYERS=2,
    #             DROPOUT=0.1,
    #             LR=1e-4,
    #             NUM_EPOCHS=15,
    #             MAX_SEQ_LEN=8,
    #             DECAY_ALPHA=1.0
    #         )
    #     ),
    # },
    {
        "name": "Transformer_Recommender",
        "recommender": TransformerRecommender(
            SimpleNamespace(
                BATCH_SIZE=128,
                EMBED_DIM=256,
                NUM_HEADS=8,
                NUM_LAYERS=5,
                DROPOUT=0.1,
                LR=1e-3,
                NUM_EPOCHS=9,
                MAX_SEQ_LEN=8,
            )
        ),
    }
]


top_k_list = [1, 3, 5]

for config in recommender_configs:
    recommender_name = config["name"]
    recommender = config["recommender"]

    print(f"\n=== Training {recommender_name} ===")

    train_data = recommender.user_item2idx(train_df.copy(), "user_id", "item")

    set_seed(42)

    recommender.train(train_data)

    print(f"--- Evaluating {recommender_name} ---")
    for top_k in top_k_list:
        recommender.evaluate(val_df, top_k=top_k)

