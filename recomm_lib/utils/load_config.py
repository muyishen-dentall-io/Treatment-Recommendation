from recomm_lib.recommender import (
    Static_Recommender,
    LLM_Recommender,
    Next_Prediction_Recommender,
    Status_Recommender,
    NCF_Recommender, 
    NCF_Recommender_GNNemb
)

from types import SimpleNamespace

import yaml

RECOMMENDER_CLASSES = {
    'Static_Recommender': Static_Recommender,
    'LLM_Recommender': LLM_Recommender,
    'Next_Prediction_Recommender': Next_Prediction_Recommender,
    'Status_Recommender': Status_Recommender,
    'NCF_Recommender': NCF_Recommender,
    'NCF_Recommender_GNNemb': NCF_Recommender_GNNemb,
}

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def instantiate_recommender(config):
    cls = RECOMMENDER_CLASSES[config['recommender_class']]
    params = config.get('params', {})

    return {
        "name": config["name"],
        "recommender": cls(SimpleNamespace(**params))
    }