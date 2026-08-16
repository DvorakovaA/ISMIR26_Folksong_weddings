"""
Feature selection from parquet bertopic probabilities output.
Using L1-penalized logistic regression to select top-K discriminative 
topics between two classes. 
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel


def get_topic_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("topic_") and c[6:].isdigit()],
        key=lambda c: int(c[6:])
    )

def load_data(parquet_path: str, labels_path, id_col: str, label_col: str, 
              class_a: str, class_b: str) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """
    Load parquet, join language labels
    """
    df = pd.read_parquet(parquet_path)

    ext = Path(labels_path).suffix.lower()
    labels_df = (
        pd.read_parquet(labels_path) if ext == ".parquet"
        else pd.read_csv(labels_path, dtype={id_col: str}, sep="\t" if ext == ".tsv" else ",")
    )
    if label_col in df.columns:
        df = df.drop(columns=[label_col])
    df = df.merge(labels_df[[id_col, label_col]], on=id_col, how="inner")

    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found. \n Available columns: {list(df.columns)}")

    available = df[label_col].unique().tolist()
    for cls in (class_a, class_b):
        if cls not in available:
            raise ValueError(f"Class '{cls}' not found in '{label_col}' \n Available: {available}")

    mask = df[label_col].isin([class_a, class_b])
    df_sub = df[mask].copy().reset_index(drop=True)

    topic_cols = get_topic_cols(df_sub)
    if not topic_cols:
        raise ValueError("No topic_* columns found in parquet.")

    X = df_sub[topic_cols].to_numpy(dtype=np.float32)
    y = (df_sub[label_col] == class_b).astype(int).to_numpy()  # 0=class_a, 1=class_b

    print(f"Loaded {len(df_sub):,} documents ({class_a}: {(y==0).sum()}, {class_b}: {(y==1).sum()})")
    print(f"Feature dimensions: {X.shape[1]} topics")

    return X, y, topic_cols, df_sub


def select_features_l1(X: np.ndarray,y: np.ndarray, topic_cols: list[str], top_k: int = 10,
                       Cs: list[float] | None = None, random_state: int = 41) -> tuple[pd.DataFrame, float, float]:

    if Cs is None:
        Cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    print(f"\nSearching C values:")
    best_C, best_score, best_coef = None, -np.inf, None

    for C in Cs:
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=C,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
        model.fit(X_scaled, y)

        selector = SelectFromModel(model, prefit=True)
        n_selected = selector.get_support().sum()

        auc = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc").mean()
        print(f"  C={C:<6}  AUC={auc:.4f}  selected by threshold={n_selected}")

        if n_selected >= top_k and auc > best_score:
            best_score = auc
            best_C = C
            best_coef = model.coef_[0].copy()

    if best_C is None:
        print(f"\nWarning: no C retained >= {top_k} features; using largest C.")
        best_C = Cs[-1]
        model = LogisticRegression(
            penalty="l1", solver="liblinear", C=best_C,
            class_weight="balanced", max_iter=1000, random_state=random_state,
        )
        model.fit(X_scaled, y)
        best_coef = model.coef_[0].copy()

    print(f"\nSelected C={best_C}  (CV AUC={best_score:.4f})")

    topic_ids = [int(c[6:]) for c in topic_cols]
    top_features = (
        pd.DataFrame({
            "topic_col": topic_cols,
            "topic_id": topic_ids,
            "coefficient": best_coef,
            "abs_coefficient": np.abs(best_coef),
        })
        .query("abs_coefficient > 0")
        .nlargest(top_k, "abs_coefficient")
        .reset_index(drop=True)
        .assign(rank=lambda d: range(1, len(d) + 1))
        .assign(favours=lambda d: d["coefficient"].apply(
            lambda c: "class_b" if c > 0 else "class_a"
        ))
    )
    return top_features, best_C, best_score

def feature_distributions(X: np.ndarray, y: np.ndarray, topic_cols: list[str], top_features: pd.DataFrame,
                          class_a: str, class_b: str) -> pd.DataFrame:
    """
    For each selected feature, compute mean and std probability in each class,
    and the absolute mean difference — useful for interpreting direction.
    """
    rows = []
    for _, feat in top_features.iterrows():
        col_idx = topic_cols.index(feat["topic_col"])
        vals_a = X[y == 0, col_idx]
        vals_b = X[y == 1, col_idx]
        rows.append({
            "rank": int(feat["rank"]),
            "topic_col": feat["topic_col"],
            "topic_id": int(feat["topic_id"]),
            "coefficient": round(float(feat["coefficient"]), 6),
            "favours": feat["favours"].replace("class_a", class_a).replace("class_b", class_b),
            f"mean_prob_{class_a}": round(float(vals_a.mean()), 6),
            f"std_prob_{class_a}": round(float(vals_a.std()), 6),
            f"mean_prob_{class_b}": round(float(vals_b.mean()), 6),
            f"std_prob_{class_b}": round(float(vals_b.std()), 6),
            "mean_diff (b - a)": round(float(vals_b.mean() - vals_a.mean()), 6),
        })
    return pd.DataFrame(rows)



def write_report(out_path: str, class_a: str, class_b: str, top_k: int, best_C: float,
                 best_score: float, top_features: pd.DataFrame, dist_df: pd.DataFrame,
                 topic_label_map: dict | None):
    lines = []
    lines.append("------------------------------")
    lines.append(f"L1 FEATURE SELECTION: '{class_a}'  vs  '{class_b}'")
    lines.append("------------------------------")
    lines.append(f"Top-K requested : {top_k}")
    lines.append(f"Best C selected : {best_C}")
    lines.append(f"CV AUC (5-fold) : {best_score:.4f}")
    lines.append("")
    lines.append(f"Top {top_k} discriminative topics:")
    lines.append("------------------------------")

    for _, row in dist_df.iterrows():
        tid = int(row["topic_id"])
        label = topic_label_map.get(tid, "") if topic_label_map else ""
        lines.append(f"\nRank {int(row['rank'])}: topic_{tid}  ({label})")
        lines.append(f"  L1 coefficient : {row['coefficient']:+.6f}  -> favours: {row['favours']}")
        lines.append(f"  Mean prob in '{class_a}' : {row[f'mean_prob_{class_a}']:.6f} ± {row[f'std_prob_{class_a}']:.6f}")
        lines.append(f"  Mean prob in '{class_b}' : {row[f'mean_prob_{class_b}']:.6f} ± {row[f'std_prob_{class_b}']:.6f}")
        lines.append(f"  Mean diff (b-a)          : {row['mean_diff (b - a)']:+.6f}")

    lines.append("\n" + "------------------------------------------------")
    report = "\n".join(lines)
    print(report)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Also save machine-readable CSV alongside the txt
    csv_path = str(Path(out_path).with_suffix(".csv"))
    dist_df.to_csv(csv_path, index=False)
    print(f"\nReport: {out_path}")
    print(f"CSV: {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="L1 feature selection between two BERTopic classes.")
    p.add_argument("parquet", help="Path to probabilities parquet.")
    p.add_argument("--labels", default=None, help="Optional external labels CSV/parquet.")
    p.add_argument("--id-col", default="id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--class-a", required=True, help="First class name.")
    p.add_argument("--class-b", required=True, help="Second class name.")
    p.add_argument("--top-k", type=int, default=10, help="Number of top features to report.")
    p.add_argument("--output", default="feature_selection.txt")
    p.add_argument("--seed", type=int, default=41)
    return p.parse_args()


def main():
    args = parse_args()

    X, y, topic_cols, df_sub = load_data(
        parquet_path=args.parquet,
        labels_path=args.labels,
        id_col=args.id_col,
        label_col=args.label_col,
        class_a=args.class_a,
        class_b=args.class_b,
    )

    top_features, best_C, best_score = select_features_l1(
        X, y, topic_cols, top_k=args.top_k, random_state=args.seed,
    )

    # Build topic_id -> topic_label map if column exists
    topic_label_map = None
    if "topic_label" in df_sub.columns and "assigned_topic" in df_sub.columns:
        topic_label_map = dict(zip(
            df_sub["assigned_topic"].astype(int),
            df_sub["topic_label"],
        ))

    dist_df = feature_distributions(X, y, topic_cols, top_features, args.class_a, args.class_b)

    write_report(
        out_path=args.output,
        class_a=args.class_a,
        class_b=args.class_b,
        top_k=args.top_k,
        best_C=best_C,
        best_score=best_score,
        top_features=top_features,
        dist_df=dist_df,
        topic_label_map=topic_label_map,
    )


if __name__ == "__main__":
    main()