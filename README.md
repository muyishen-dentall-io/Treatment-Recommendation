# Dental Treatment Recommendation System

## 1. Introduction

It is important to help clinics identify patients who are likely to bring higher long-term value. However, accurately estimating a patient’s Lifetime Value (LTV) is challenging due to limited and noisy financial records. To address this, the project designs a suite of treatment recommendation algorithms to bypass direct LTV prediction. By observing which patients receive recommendations for higher-value treatments, clinics can more effectively focus on and invest in high-value patients.

---

## 2. Project Overview

This repository is organized into several core components. Each folder has a dedicated README with further details.

| Folder / File         | Description                                                                                  |
|-----------------------|---------------------------------------------------------------------------------------------|
| [`configs/`](./configs)                 | YAML config files for running different recommenders and experiments. [Read more &rarr;](./configs/README.md)  |
| [`data/`](./data)                       | Processed data files: patient info, treatments, splits, prompts. [Read more &rarr;](./data/README.md)         |
| [`emb_generate/`](./emb_generate)       | Embedding generation code: GNN/LLM models, visualizations. [Read more &rarr;](./emb_generate/README.md)        |
| [`recomm_lib/`](./recomm_lib)           | Main recommendation system implementations and utilities. [Read more &rarr;](./recomm_lib/README.md)           |                                             |
| [`train.py`](./train.py)                | Main training and evaluation script.                                                                         |

> **Note:** Each major subfolder contains a more detailed `README.md` describing usage and implementation details.

---

## 3. Benchmark Performance

This project implements and benchmarks several types of recommenders for dental treatment prediction:

- **Next-Item Recommender:** A Transformer-based sequential model that predicts the next likely treatments based on a patient’s historical sequence.
- **Status Recommender:** Uses LLM-generated treatment descriptions, encodes them with a text encoder, and applies a Transformer over the patient’s history to produce a status embedding, which is compared with candidate treatment embeddings.
- **NCF + GNN Embeddings:** Neural Collaborative Filtering model enhanced with item (treatment) embeddings pre-trained via a Graph Neural Network on the treatment co-occurrence graph.
- **LLM (GPT4.1) - Treatment:** Uses a large language model (LLM) to generate context-aware treatment recommendations based only on structured treatment history.
- **LLM (GPT4.1) - Treatment + Comment:** Similar to the above, but also includes additional free-text clinical comments as input.
- **Neural Collaborative Filtering (NCF):** Trains patient and treatment embeddings with a neural network and recommends based on their similarity.
- **Static Baseline:** Always recommends the most frequent treatments in the training data, without personalization.

---

### Dataset

- **Source:** Anonymized real-world dental clinic records.
- **Task:** Given a patient's history, predict which treatments are most likely next.
- **Example:** Each row represents a treatment event for a patient.
- **Columns:**  
    | user_id | item | date       | tooth_position | gender | birthday   | smoking | comment
    |------------|---------------|------------|--------|--------|------------|---------|---------|
    | 7       | 34001C根尖周 X光攝影     | 2018-06-28 | 17 |FEMALE      | 1922-07-20 | 0      | CC:Asking for dental check up... |
    | 7       | 90015C根管開擴及清創     | 2018-07-05 | 24 |FEMALE      | 1922-07-20 | 0      | RD isolation,access opening,...
    | 13       | 92001C非特定局部治療     | 2018-01-15 | 46 | FEMALE      | 1926-03-03 | 0      | Oral ulcer on R't,L't cheek mucosa... |
    | ... | ... | ... | ... | ... | ... | ... | ... |


### Metrics

Evaluation focuses on **Recall@5, Precision@5, HitRate@5, and MRR@5**.

| Model                    | Recall@5 | Precision@5 | HitRate@5 | MRR@5 |
|----------------------------|:----------:|:-------------:|:-----------:|:-------:|
| Next-Item  Recommender  | **0.6318**     | **0.4120**        | 0.9151      | **0.8571**  |
| Status Recommender       | 0.6298     | 0.4066        | **0.9168**      | 0.7968  |
| NCF + GNN Embeddings     | 0.5846     | 0.3786        | 0.8964      | 0.8308  |
| LLM (GPT4.1) - Treatment | 0.5806     | 0.3692        | 0.8878      | 0.8017  |
| Neural Collaborative Filtering (NCF)          | 0.5678  | 0.3601 | 0.8967      | 0.7353   |
| LLM (GPT4.1) - Treatment + Comment          | 0.5304     | 0.3339        | 0.7979      | 0.7266  |
| Static Baseline          | 0.2536     | 0.3377        | 0.8645      | 0.7814  |

*Note: Actual numbers may vary slightly depending on hardware and random seeds.*

---

## 4. Get Started

### 1. Set Your OpenAI API Key

Open the `.env` file in the root folder and set your OpenAI API key:
```
OPENAI_API_KEY='sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
```

### 2. Create and Activate a Virtual Environment

Create a virtual environment under your home directory (e.g., `~/.recommend_env`):

``` bash
python -m venv ~/.recommend_env
source ~/.recommend_env/bin/activate
```

> On Windows, use `~\.recommend_env\Scripts\activate` instead of the `source` command above.


### 3. Install Dependencies

First, install PyTorch and related packages:

``` bash
pip install torch torchvision torchaudio
```

Then install the remaining requirements:

``` bash
pip install -r requirements.txt
```

### 4. Generate Pretrained Embeddings

In this project, the Status Recommender utilizes text embeddings generated by language models, while NCF can be augmented with Graph Neural Network (GNN) pretrained item embeddings.  
To generate these embeddings, run the scripts inside the `./emb_generate` folder:

- **To generate GNN-based embeddings (Used by NCF_GNN_Recommender):**
    ``` bash
    cd emb_generate/gnn
    python run_gnn.py
    ```
- **To generate language model text embeddings (Used by Status_Recommender):**
    ``` bash
    cd emb_generate/llm
    python generate_descriptions.py
    python generate_embeddings.py
    ```

## 5. Model Training & Evaluation

All models are configured using YAML files inside the `./configs` folder, where you can specify training hyperparameters such as the number of epochs, learning rate, network architecture, and more.

Example:
```yaml
# next_pred.yaml
name: Next_Prediction_Recommender
recommender_class: Next_Prediction_Recommender
params:
  BATCH_SIZE: 128
  EMBED_DIM: 128
  NUM_HEADS: 2
  NUM_LAYERS: 2
  DROPOUT: 0.1
  LR: 0.0001
  NUM_EPOCHS: 9
  MAX_SEQ_LEN: 8
  DECAY_ALPHA: 1.0
```

In addition, the file `./configs/run_list.yaml` allows you to specify a list of models to train and evaluate in batch. Simply uncomment the desired models in the YAML file.

**Example:**
```yaml
# run_list.yaml
recommenders:
    # Uncomment to enable training for these models
    # - static.yaml
    # - ncf.yaml
    # - ncf_gnnemb.yaml
    - next_pred.yaml
```

To train and evaluate the models uncommented in `run_list.yaml`, simply run:

```bash
python train.py
```

from the root folder.

After training, the evaluation logs containing the metrics (Top-1, Top-3, Top-5, etc.) will be saved into the `./results` folder.
