import pandas as pd
import sqlite3

def load_treatment_data(DB_PATH):
    
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("""
            SELECT patient_id AS user_id, treatment_item AS item, date
            FROM dental_treatments
            WHERE treatment_item IS NOT NULL
        """, conn)
    
    df.drop_duplicates(inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    
    return df


def filter_data(df, MIN_USER_INTERACTIONS=2, MIN_ITEM_INTERACTIONS=2):
    user_counts = df["user_id"].value_counts()
    item_counts = df["item"].value_counts()
    
    print(df['user_id'].nunique())
    
    before_user = df['user_id'].nunique()
    before_item = df['item'].nunique()
    before_inter = len(df)
     
    df = df[df["user_id"].isin(user_counts[user_counts >= MIN_USER_INTERACTIONS].index)]
    df = df[df["item"].isin(item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index)]

    print(f"🧍 Users before/after filtering: {before_user, df['user_id'].nunique()}")
    print(f"🦷 Items before/after filtering: {before_item, df['item'].nunique()}")
    print(f"🔗 Interactions before/after filtering: {before_inter, len(df)}")
    
    return df