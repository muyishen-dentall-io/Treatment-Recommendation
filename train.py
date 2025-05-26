import pdb
import pandas as pd
import os
import yaml

from recomm_lib import (
    server_data_to_db,
    load_treatment_data,
    train_test_split_by_n_year,
    set_seed,
    load_config, 
    instantiate_recommender
)

def main():
    # Step 0.1: Run server_data_to_db if the server data has not saved to the local db
    # server_data_to_db(DB_PATH='./data/clinic_data.db', n_patient=10000)

    # Step 0.2: Execute SQL command to extract needed info from DB to a dataframe
    # df = load_treatment_data(
    #     './data/clinic_data.db',
    #     save_to_csv=True,
    #     save_folder='./data/'
    # )

    # Step 0.3: Load if the above dataframe has been saved to a csv file
    # df = pd.read_csv('./data/patient_data.csv')

    # Step 0.4: Apply train test split and save to different csv files
    # train_df, val_df = train_test_split_by_n_year(df, save_to_csv=True, save_folder="./data/")

    # --------------------------------------------------------------------------------------------
    # Uncomment the above code for the first execution of the program
    # --------------------------------------------------------------------------------------------

    train_df = pd.read_csv("./data/train_df.csv")
    val_df = pd.read_csv("./data/val_df.csv")

    with open('./configs/run_list.yaml', 'r') as f:
        run_list = yaml.safe_load(f)

    recommender_config_files = [os.path.join('./configs', fname) for fname in run_list['recommenders']]

    recommender_configs = []
    for config_path in recommender_config_files:
        config = load_config(config_path)
        recommender_configs.append(instantiate_recommender(config))


    top_k_list = [1, 3, 5]

    for config in recommender_configs:
        recommender_name = config["name"]
        recommender = config["recommender"]

        print(f"\n=== Training {recommender_name} ===")

        train_data = recommender.user_item2idx(train_df.copy())

        set_seed(42)

        recommender.train(train_data)

        print(f"--- Evaluating {recommender_name} ---")
        for top_k in top_k_list:
            recommender.evaluate(val_df, top_k=top_k, save_log=True)

if __name__ == "__main__":
    main()