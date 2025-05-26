import pandas as pd
import os


def save_train_val_csv(train_df, val_df, save_folder="./data/"):
    train_csv_path = save_folder + "train_df.csv"
    val_csv_path = save_folder + "val_df.csv"
    os.makedirs(os.path.dirname(train_csv_path), exist_ok=True)
    import csv

    train_df.to_csv(train_csv_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Training Data successfully saved into path {train_csv_path}")
    val_df.to_csv(val_csv_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Validation Data successfully saved into path {val_csv_path}")


def train_test_split_by_ratio(df, ratio=0.2, save_to_csv=True, save_folder="./data/"):
    train_rows = []
    val_rows = []

    for user_id, group in df.sort_values("date").groupby("user_id"):
        n = len(group)
        val_count = max(1, int(n * ratio))
        train_rows.append(group.iloc[:-val_count])
        val_rows.append(group.iloc[-val_count:])

    train_df = pd.concat(train_rows)
    val_df = pd.concat(val_rows)

    print(f"📚 Training interactions: {len(train_df)}")
    print(f"🧪 Validation interactions (before filtering): {len(val_df)}")

    if save_to_csv:
        save_train_val_csv(train_df, val_df, save_folder)

    return train_df, val_df


def train_test_split_by_n_year(
    df, n_year=1, min_span_years=2, save_to_csv=True, save_folder="./data/"
):
    train_rows = []
    val_rows = []

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    total_user = 0
    n_span_user = 0

    for user_id, group in df.sort_values("date").groupby("user_id"):

        total_user += 1

        first = group["date"].min()
        last = group["date"].max()
        span_years = (last - first).days / 365.25

        if span_years < min_span_years:
            train_rows.append(group)
            n_span_user += 1
            continue

        val_start = last - pd.DateOffset(years=n_year)
        train_part = group[group["date"] < val_start]
        val_part = group[group["date"] >= val_start]

        if len(val_part) == 0:
            train_rows.append(group)
        else:
            train_rows.append(train_part)
            val_rows.append(val_part)

    train_df = pd.concat(train_rows)
    val_df = pd.concat(val_rows)

    print(f"📚 Training interactions: {len(train_df)}")
    print(f"🧪 Validation interactions (after span filtering): {len(val_df)}")

    print(
        "Total User: {}, Spanning User: {}, Percentage: {:.4f}".format(
            total_user,
            total_user - n_span_user,
            (total_user - n_span_user) / total_user,
        )
    )

    if save_to_csv:
        save_train_val_csv(train_df, val_df, save_folder)

    return train_df, val_df


def filter_treatment(df, include_treatment_file):
    orig_len = len(df)

    with open(include_treatment_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    df = df[df["item"].isin(lines)]
    cur_len = len(df)

    print(f"Interactions before/after filtering: {orig_len, cur_len}")

    return df
