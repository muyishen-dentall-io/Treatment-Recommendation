# Embedding Generation

This folder contains all scripts and modules for generating treatment/item embeddings used by the recommenders in this project.  
There are two main approaches provided:

- **Graph Neural Network (GNN) embeddings** for capturing treatment co-occurrence relationships.
- **Large Language Model (LLM) embeddings** for capturing semantic similarity between treatment descriptions.

---

**Embedding Output Format**

- All generated embeddings are saved in JSON format:
    - For GNN: The JSON maps each treatment name to its embedding vector.
    - For LLM: The JSON maps each treatment name to a list of embedding vectors (if `N_DESCRIPTIONS` in `config.py` is greater than 1).

> **Note:** By default, the NCF_GNN Recommender and Status Recommender will automatically search for their respective embedding files in these folders (`emb_generate/gnn` for GNN embeddings and `emb_generate/llm` for LLM embeddings).  

> Please ensure that the embeddings are correctly generated and available in the appropriate directory for your chosen model to function properly.

---

## GNN Embeddings

This module trains a Graph Neural Network on the treatment co-occurrence graph, and outputs embedding vectors for each treatment item.  
These embeddings can be used to enhance collaborative filtering recommenders such as NCF.

By default, the GNN module uses the training dataframe located at `../../data/train_df.csv` (from the project root) and constructs the co-occurrence graph based on treatments that appear together in patient histories.

**Main scripts:**

- `run_gnn.py` — Entry point for training and exporting GNN embeddings.
- `gnn_lib/` — Core code for building, training, and visualizing the GNN.
    - `model.py` — GNN architecture definition.
    - `dataset.py` — Prepares the co-occurrence graph data.
    - `train.py` — Training logic.
    - `visualize.py` — Tools for embedding visualization (e.g., t-SNE plots).

**Usage Example:**

```bash
cd emb_generate/gnn
python run_gnn.py
```

This will output the learned treatment embeddings (and optional visualizations) to the same folder.

---

## LLM Embeddings

This module uses a large language model (LLM) to generate text-based embeddings for treatment descriptions.  
These are mainly used by the Status Recommender, but can be useful in other settings as well.

The parameters for this module are set in `config.py`, including:

- `OPENAI_MODEL`: The LLM to use for generating descriptions and embeddings (default: `"gpt-4.1"`).
- `N_DESCRIPTIONS`: Number of different descriptions to generate per treatment (default: `5`).
- `OUT_DIM`: Output dimension for embedding vectors (default: `128`).

**Main scripts:**

- `generate_descriptions.py` — Generates text descriptions for each treatment.
- `generate_embeddings.py` — Uses a language model to embed each description as a fixed-length vector.
    - Supports two embedding backends via the `--method` argument:
        - `sbert`: Uses SentenceTransformer (multilingual-e5-base).
        - `openai`: Uses OpenAI API (e.g., text-embedding-3-large).
- `config.py` — Model and data settings.

**Usage Example:**
```bash
cd emb_generate/llm
python generate_descriptions.py
python generate_embeddings.py --method openai
```

The output will be a JSON file containing the embedding vectors for all treatments.

---

## Notes

- Make sure your data files (`item_list.txt` and relevant treatment information) are up to date before generating embeddings.
- The generated embeddings can be found in the output of each respective subfolder.
- Visualizations (such as `treatment_embeddings_visualization.html`) are also available for qualitative inspection.

---
