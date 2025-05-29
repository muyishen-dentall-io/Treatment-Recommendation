# Recommendation Library

This folder contains all the core modules for implementing, extending, and running different recommendation models in this project. It is organized by functionality to make it easy to modify and add new recommenders.

## Submodules

### `data/`
- **Purpose:** Data loading and preprocessing utilities.
    - `load_data_from_db.py`: Extracts and prepares data from the local clinic database. (By default, `load_data_from_server` is called first to create the local database if it does not already exist.)
    - `load_data_from_server.py`: Used for loading data directly from an external server if needed. If you update the server with data from a different clinic, please rerun this function (see train.py for details).
    - `data_split.py`: Functions for splitting the dataset into training and validation sets. By default, it assigns each patient’s last year of data to the validation set and the remaining records to the training set.

### `recommender/`
- **Purpose:** All model implementations for recommendation.
    - `llm_recommender.py`: Recommender that uses a predefined query prompt along with the patient’s history to generate responses via an LLM. By default, the recommender uses the OpenAI API to obtain responses from GPT-4.1; note that an interface for Ollama is also implemented.
    - `ncf_recommender.py`: Neural Collaborative Filtering model that learns patient and treatment embeddings, and incorporates gender as an input feature. The model combines these three embedding vectors to predict a relevance score for each treatment, and then recommends the top-K scoring treatments to the patient.
    - `next_prediction_recommender.py`: Transformer-based sequential model for next-treatment prediction. Note that the output dimension is fixed to the set of treatment classes observed during training.
    - `static_recommender.py`: Simple baseline that always recommends the most common treatments in the training data.
    - `status_recommender.py`: Recommender that uses a Transformer to generate a status embedding from the patient’s history. Treatments are recommended by selecting the top-K with the highest cosine similarity between the status embedding and treatment embeddings. Note that treatment embeddings can be either fixed random vectors or derived from treatment text embeddings.

### `utils/`
- **Purpose:** Helper utilities for reproducibility and configuration.
    - `set_seed.py`: Ensures deterministic results by setting random seeds.
    - `load_config.py`: Loads model and training configuration files.

## How to Add a New Recommender

1. **Create a new Python module** (e.g., `my_custom_recommender.py`) in the `recommender/` subfolder.  
   Implement the following functions:
   - `user_item2idx`: Maps patient and treatment identifiers to numeric indices.
   - `train`: Defines the training logic for your model.
   - `evaluate`: Handles model evaluation; refer to other recommenders for standard metric implementations.

2. **Import your new recommender** in both the `__init__.py` file of the `recommender/` submodule and the top-level `recomm_lib` module.

3. **Register your recommender** in `utils/load_config.py`, add a corresponding YAML config to the `/configs` folder, and list it in `run_list.yaml`.

4. **Run `train.py`** to verify that your new model integrates and trains successfully.

---
