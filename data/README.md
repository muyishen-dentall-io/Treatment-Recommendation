# Data Folder

This folder contains all the data files required or generated for training and evaluating the recommendation models.

## File Descriptions

| File Name            | Description                                                                              |
|----------------------|------------------------------------------------------------------------------------------|
| `item_list.txt`      | List of all treatment item names (one per line).    |
| `predict_prompt.txt` | Prompt template used for LLM-based recommenders.                                         |
| `clinic_data.db`     | Anonymized clinic database (SQLite format) containing raw patient and treatment records. |
| `patient_data.csv`   | CSV file of processed patient and treatment information extracted from the database.  |
| `train_df.csv`       | CSV file containing the training data split.                                             |
| `val_df.csv`         | CSV file containing the validation data split.                                           |

## Sample Data Preview

**patient_data.csv** (or `train_df.csv`, `val_df.csv`)
| user_id | item              | date       | tooth_position | gender | birthday   | smoking | comment                              |
|---------|-------------------|------------|---------------|--------|------------|---------|--------------------------------------|
| 7       | 34001C根尖周 X光攝影 | 2018-06-28 | 17            | FEMALE | 1922-07-20 | 0       | CC:Asking for dental check up...     |
| 7       | 90015C根管開擴及清創 | 2018-07-05 | 24            | FEMALE | 1922-07-20 | 0       | RD isolation, access opening,...     |
| 13      | 92001C非特定局部治療 | 2018-01-15 | 46            | FEMALE | 1926-03-03 | 0       | Oral ulcer on R't, L't cheek mucosa...|

*(For `clinic_data.db`, you can explore its content using DB Browser for SQLite or any database client.)*


## Provided Dataset Details

The dataset provided in this repository consists of:

- **Training set:** 90,912 treatment records (`train_df.csv`)
- **Validation set:** 16,733 treatment records (`val_df.csv`)

The dataset is split at the **patient level**:  
For each patient, all treatment records from their **last year** of history are assigned to the validation set (`val_df.csv`). The remaining earlier records are used for training (`train_df.csv`). This simulates a real-world scenario where, given a patient’s historical treatments, we predict which treatments they are likely to receive in the following year.


## Data Processing Workflow

The first two files (`item_list.txt` and `predict_prompt.txt`) are the only files that should **originally** be in this folder.  
Other files (`clinic_data.db`, `patient_data.csv`, `train_df.csv`, and `val_df.csv`) are **generated** by running the commented-out preprocessing section in `train.py`.

**Preprocessing steps (Please refer to `train.py` for detail):**  
- Use `server_data_to_db` to fetch server data and save it as `clinic_data.db`.
- Run `load_treatment_data` to extract relevant data and save as `patient_data.csv`.
- Apply `train_test_split_by_n_year` to split the dataset into `train_df.csv` and `val_df.csv`.

We provide the generated database and CSV files here for your convenience, allowing you to skip the preprocessing steps and get started quickly.

**If you add data from another clinic:**
- Update `item_list.txt` to include all treatment items in the new dataset.
- Rerun the preprocessing pipeline in `train.py` (uncomment the relevant sections) to generate new `clinic_data.db`, `patient_data.csv`, `train_df.csv`, and `val_df.csv` files.

## Notes

- `item_list.txt` and `predict_prompt.txt` are required for all training runs.
- LLM-based recommenders use `predict_prompt.txt` for prompt construction.
- Ensure `item_list.txt` is always up to date with the treatments in your database to avoid errors.

For more details on data format and preprocessing, see the comments in `train.py`.

---
