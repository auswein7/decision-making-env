# run this module through CLI: "/usr/local/bin/python3.11 csv_postprocessor.py" from this directory

import os
import pandas as pd

CSV_PATH = "/Users/austinweingart/Documents/GradSchool/thesis/decision-making-env/app/out/system_trials.csv"
SUMMARY_PATH = "out/system_summary.csv"
PARAM_PATH = "out/system_params.csv"

def load_trial_data(csv_path=CSV_PATH):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    return pd.read_csv(csv_path)

def summarize_best_scores(df):
    return (
        df.groupby("system_id")["observed_optimal"]
        .max()
        .reset_index()
        .rename(columns={"observed_optimal": "best_observed_score"})
    )

def compute_average_gap(df):
    return (
        df.groupby("system_id")["gap_to_local_min"]
        .mean()
        .reset_index()
        .rename(columns={"gap_to_local_min": "avg_gap_to_local_min"})
    )

def extract_local_minima(df):
    return (
        df.groupby("system_id")["local_minima"]
        .first()
        .reset_index()
        .rename(columns={"local_minima": "local_minima_value"})
    )

def summarize_best_scores_by_param(df):
    """
    Returns the best score per (system_id, distribution, param_value).
    """
    summary = (
        df.groupby(["system_id", "distribution", "param_value"])["observed_optimal"]
        .max()
        .reset_index()
        .rename(columns={"observed_optimal": "best_score_for_param"})
        .sort_values(["system_id", "distribution", "param_value"])
    )
    return summary

def filter_best_param_per_distribution(param_df):
    """
    For each (system_id, distribution), keep only the param_value with the highest score.
    """
    return (
        param_df.sort_values(["system_id", "distribution", "best_score_for_param"], ascending=[True, True, False])
        .groupby(["system_id", "distribution"])
        .head(1)
        .reset_index(drop=True)
    )

def export_summary(summary_df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    summary_df.to_csv(path, index=False)
    print(f"Summary exported to: {path}")

def main():
    df = load_trial_data()

    best_scores = summarize_best_scores(df)
    avg_gaps = compute_average_gap(df)
    local_min = extract_local_minima(df)
    best_by_param = summarize_best_scores_by_param(df)
    best_param_per_dist = filter_best_param_per_distribution(best_by_param)

    merged_summary = pd.merge(best_scores, avg_gaps, on="system_id", how="outer")
    merged_summary = pd.merge(merged_summary, local_min, on="system_id", how="outer")

    export_summary(merged_summary, path=SUMMARY_PATH)
    export_summary(best_param_per_dist, path=PARAM_PATH)

if __name__ == "__main__":
    main()
