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
| [`recomm_lib/`](./recomm_lib)           | Main recommendation system implementations and utilities. [Read more &rarr;](./recomm_lib/README.md)           |
| [`results/`](./results)                 | Output metrics, predictions, and evaluation logs for each recommender. [Read more &rarr;](./results/README.md) |                                              |
| [`train.py`](./train.py)                | Main training and evaluation script.                                                                         |

> **Note:** Each major subfolder contains a more detailed `README.md` describing usage and implementation details.

---

## 3. Benchmark Performance

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
| Next-Item  Recommender  | 0.6318     | 0.4120        | 0.9151      | 0.8571  |
| Status Recommender       | 0.6298     | 0.4066        | 0.9168      | 0.7968  |
| NCF + GNN Embeddings     | 0.5846     | 0.3786        | 0.8964      | 0.8308  |
| LLM (GPT4.1) - Treatment | 0.5806     | 0.3692        | 0.8878      | 0.8017  |
| Neural Collaborative Filtering (NCF)          | 0.5678  | 0.3601 | 0.8967      | 0.7353   |
| LLM (GPT4.1) - Treatment + Comment          | 0.5304     | 0.3339        | 0.7979      | 0.7266  |
| Static Baseline          | 0.2536     | 0.3377        | 0.8645      | 0.7814  |

*Note: Actual numbers may vary slightly depending on hardware and random seeds.*

---

## 4. Get Started

### Requirements