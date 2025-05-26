import pandas as pd
import sqlite3
import os

def load_treatment_data(DB_PATH='./data/clinic_data.db', save_to_csv=True, save_folder='./data/'):
    conn = sqlite3.connect(DB_PATH)

    info_df = pd.read_sql_query(
        """
            WITH patient_info AS (
                SELECT patient_id, gender, birthday
                FROM patients
            )
            SELECT ps.patient_id, ps.smoking, pi.gender, pi.birthday
            FROM patient_status ps
            LEFT JOIN patient_info pi
            ON ps.patient_id = pi.patient_id
        """,
        conn,
    )

    df = pd.read_sql_query(
        """
            SELECT patient_id AS user_id, 
            treatment_item AS item, 
            date,
            tooth_position,
            doctor_comment as comment 
            FROM dental_treatments
            WHERE treatment_item IS NOT NULL
        """,
        conn,
    )

    df["date"] = pd.to_datetime(df["date"])

    n_df = df.merge(info_df, how="left", left_on="user_id", right_on="patient_id")

    n_df = n_df.drop(columns=["patient_id"])

    if save_to_csv:
        csv_path = save_folder + 'patient_data.csv'
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        import csv
        n_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)
        print(f'Patient Data successfully saved into path {csv_path}')

    return n_df
