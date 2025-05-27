# Configs Folder

This folder contains all the YAML configuration files used to train, evaluate, and benchmark different recommendation models in this project.

## Contents

- **Model Config Files:**  
  Each YAML file (e.g., `next_pred.yaml`, `ncf.yaml`, `llm_treatment.yaml`, etc.) specifies the model type, training hyperparameters, and any relevant settings.
- **run_list.yaml:**  
  Specifies a list of models to train and evaluate in batch mode. Uncomment the desired model config files to enable them for batch training.

## Example Model Config Structure

Each config file defines:
- Model name and class
- Model-specific parameters (batch size, embedding dimension, number of layers, etc.)

**Example (`next_pred.yaml`):**
```yaml
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

Hyperparameter meanings:

- **BATCH_SIZE**: Number of samples processed in one training step.
- **EMBED_DIM**: Size of embedding vectors for model inputs.
- **NUM_HEADS**: Number of attention heads in each Transformer layer.
- **NUM_LAYERS**: Number of stacked Transformer layers.
- **DROPOUT**: Probability of an element being zeroed during dropout (regularization).
- **LR**: Initial learning rate for the optimizer.
- **NUM_EPOCHS**: Number of times the entire training dataset is passed through the model.
- **MAX_SEQ_LEN**: Maximum length of the input sequence (patient history) considered.
- **DECAY_ALPHA**: Soft label decay rate (see the technical report for details).

## Using `run_list.yaml`

To run multiple models in sequence, use `run_list.yaml`.  
Uncomment the config files for the models you wish to train/evaluate.

**Example:**
```yaml
recommenders:
  - static.yaml
  - ncf.yaml
  - ncf_gnnemb.yaml
  - next_pred.yaml
  - status.yaml
#   - llm_note.yaml
#   - llm_treatment.yaml
#   - llm_all.yaml
```
The above setting will train and evaluate all recommenders except the LLM-based models.

You can then launch batch training by running the following command in the root folder:
``` bash
python train.py
```

## List of Config Files

| File Name           | Description                       |
|---------------------|-----------------------------------|
| llm_all.yaml        | LLM recommender (all features)    |
| llm_note.yaml       | LLM recommender (clinical notes only)|
| llm_treatment.yaml  | LLM recommender (treatment only)  |
| ncf_gnnemb.yaml     | NCF with GNN-pretrained embeddings|
| ncf.yaml            | Standard Neural Collaborative Filtering|
| next_pred.yaml      | Next-item prediction (Transformer)|
| static.yaml         | Static (most-frequent) baseline   |
| status.yaml         | Status Recommender (embedding based)|
| run_list.yaml       | List for batch training           |

---