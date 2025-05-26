import pandas as pd
from collections import Counter
from itertools import combinations
import torch

def load_patient_records(path):
    train_df = pd.read_csv(path)
    patient_records = dict()
    for user_id, group in train_df.groupby('user_id'):
        patient_records[user_id] = group['item'].to_list()
    return patient_records

def build_cooccurrence_edges(patient_records):
    cooccurrence_counter = Counter()
    for treatments in patient_records.values():
        for t1, t2 in combinations(sorted(treatments), 2):
            cooccurrence_counter[(t1, t2)] += 1
            cooccurrence_counter[(t2, t1)] += 1
    return cooccurrence_counter

def build_vocab(patient_records):
    treatments = set()
    for record in patient_records.values():
        treatments.update(record)
    treatment2id = {treatment: idx for idx, treatment in enumerate(sorted(treatments))}
    id2treatment = {idx: treatment for treatment, idx in treatment2id.items()}
    return treatment2id, id2treatment
