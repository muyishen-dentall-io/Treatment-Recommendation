import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1"

ITEM_LIST_PATH = "../../data/item_list.txt"
DESCRIPTION_OUTPUT_PATH = "treatment_descriptions.json"

DESCRIPTION_EMBED_OUTPUT = "description_embeddings.json"

N_DESCRIPTIONS = 5
OUT_DIM = 128
