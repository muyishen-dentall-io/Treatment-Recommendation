import json
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import config

def embed_with_sbert(data):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    description_emb_dict = {}
    all_emb = []

    for treat, desc_list in tqdm(data.items(), desc="SBERT: Generating embeddings"):
        emb_list = []
        for desc in desc_list:
            emb = model.encode('passage: ' + treat + ' ' + desc).tolist()
            emb_list.append(emb)
            all_emb.append(emb)
        description_emb_dict[treat] = emb_list

    if all_emb:
        embeddings_np = np.array(all_emb)
        pca = PCA(n_components=config.OUT_DIM)
        pca.fit(embeddings_np)
        for treat, emb_list in tqdm(description_emb_dict.items(), desc="SBERT: Applying PCA"):
            for i in range(len(emb_list)):
                emb_list[i] = pca.transform(np.expand_dims(emb_list[i], axis=0)).squeeze().tolist()
        print(f"SBERT embeddings generated and reduced to {config.OUT_DIM} dimensions.")
    else:
        print("No embeddings generated.")
    return description_emb_dict

def embed_with_openai(data):
    from openai import OpenAI
    import os
    from dotenv import load_dotenv, find_dotenv

    _ = load_dotenv(find_dotenv())
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    description_emb_dict = {}
    for treat, desc_list in tqdm(data.items(), desc="OpenAI: Generating embeddings"):
        emb_list = []
        for desc in desc_list:
            try:
                response = client.embeddings.create(
                    input=treat + ' ' + desc,
                    model="text-embedding-3-large"
                )

                emb = response.data[0].embedding[:config.OUT_DIM]
                emb_list.append(emb)
            except Exception as e:
                print(f"❌ OpenAI error for '{treat}': {e}")
                emb_list.append([0.0]*config.OUT_DIM)
        description_emb_dict[treat] = emb_list
    print("OpenAI embeddings generated (dim=config.OUT_DIM).")
    return description_emb_dict

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["sbert", "openai"],
        default="sbert",
        help="Which embedding method to use: 'sbert' for SentenceTransformer + PCA, 'openai' for OpenAI API (default: sbert)"
    )
    args = parser.parse_args()

    with open(config.DESCRIPTION_OUTPUT_PATH, "r", encoding='utf-8') as f:
        data = json.load(f)

    if args.method == "sbert":
        description_emb_dict = embed_with_sbert(data)
    elif args.method == "openai":
        description_emb_dict = embed_with_openai(data)
    else:
        raise ValueError("Unsupported method")

    with open(config.DESCRIPTION_EMBED_OUTPUT, "w", encoding='utf-8') as f:
        json.dump(description_emb_dict, f, indent=2)
    print(f"Embeddings saved to {config.DESCRIPTION_EMBED_OUTPUT}")

if __name__ == "__main__":
    main()
