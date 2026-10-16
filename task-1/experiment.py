#!/usr/bin/env python3
"""Temporal KION recommendation experiment: manual vs generated features."""
import argparse
import hashlib
import itertools
import json
import time
import urllib.request
import zipfile
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

URL = "https://raw.githubusercontent.com/irsafilo/KION_DATASET/b242d2c781110902cbc8e5139f6fcefa64bbc845/data_en.zip"
SHA256 = "fddc5b2215bfff49bcfa21e464441f1fc6dc30f8f4afe41d3f23bf3fc45f3542"
TRAIN_CUTOFFS = ["2021-07-05", "2021-07-12", "2021-07-19", "2021-07-26"]
VAL_CUTOFF = "2021-08-02"
TEST_CUTOFF = "2021-08-09"
WINDOWS = [3, 7, 14, 30, 60, 120, 180]
RELATIONS = ["item", "user", "pair", "genre"]
MANUAL = ["item_7", "item_30", "item_180", "user_7", "user_30",
          "user_180", "pair_180", "genre_30", "genre_180",
          "user_recency", "item_recency", "genre_share_30", "genre_share_180",
          "item_trend_7_30"]
STATIC = ["release_year", "age_rating", "for_kids", "content_type",
          "genre_id", "user_age", "user_income", "user_sex", "user_kids"]
DERIVED = ["user_recency", "item_recency", "genre_share_30", "genre_share_180",
           "item_trend_7_30"]
CATEGORICAL = ["content_type", "genre_id", "user_age", "user_income", "user_sex", "user_kids"]


def load_data(zip_path, n_users, seed):
    if not zip_path.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, zip_path)
    with zip_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError(f"Unexpected data archive checksum: {digest}")
    with zipfile.ZipFile(zip_path) as z:
        with z.open("data_en/interactions.csv") as f:
            ev = pd.read_csv(f, usecols=["user_id", "item_id", "last_watch_dt"])
        with z.open("data_en/items_en.csv") as f:
            items = pd.read_csv(f, usecols=["item_id", "content_type", "release_year",
                                               "genres", "for_kids", "age_rating"])
        with z.open("data_en/users_en.csv") as f:
            users = pd.read_csv(f, usecols=["user_id", "age", "income", "sex", "kids_flg"])
    ev["date"] = pd.to_datetime(ev.pop("last_watch_dt"))
    pool = np.sort(ev.loc[ev.date < TRAIN_CUTOFFS[0], "user_id"].unique())
    chosen = np.random.default_rng(seed).choice(pool, size=min(n_users, len(pool)), replace=False)
    ev = ev[ev.user_id.isin(chosen)].copy()
    items["genre"] = items.genres.fillna("unknown").str.split(", ").str[0]
    items = items.drop(columns="genres").drop_duplicates("item_id")
    items["genre_id"] = pd.Categorical(items.genre).codes
    users = users[users.user_id.isin(chosen)].drop_duplicates("user_id")
    for frame, columns in [(items, ["content_type"]), (users, ["age", "income", "sex"])]:
        for col in columns:
            frame[col] = pd.Categorical(frame[col].fillna("unknown")).codes
    ev = ev.merge(items[["item_id", "genre"]], on="item_id", how="left")
    ev["genre"] = ev.genre.fillna("unknown")
    return ev, items, users, len(pool)


def lookup_counts(history, relation, keys, days, cutoff):
    recent = history[history.date >= cutoff - pd.Timedelta(days=days)]
    columns = {"item": ["item_id"], "user": ["user_id"],
               "pair": ["user_id", "item_id"], "genre": ["user_id", "genre"]}[relation]
    counts = recent.groupby(columns, observed=True).size()
    if len(columns) == 1:
        return pd.Series(keys[columns[0]]).map(counts).fillna(0).to_numpy(dtype=np.float32)
    index = pd.MultiIndex.from_frame(keys[columns])
    return counts.reindex(index, fill_value=0).to_numpy(dtype=np.float32)


def make_snapshot(ev, items, users, cutoff_string, n_candidates):
    t0 = time.monotonic()
    cutoff = pd.Timestamp(cutoff_string)
    history = ev[ev.date < cutoff]
    future = ev[(ev.date >= cutoff) & (ev.date < cutoff + pd.Timedelta(days=7))]
    active_users = np.sort(history.user_id.unique())
    popularity_order = history.item_id.value_counts().index.to_numpy()
    seen_by_user = history.groupby("user_id").item_id.agg(set)
    def unseen_for_user(user):
        seen = seen_by_user.loc[user]
        return list(itertools.islice((item for item in popularity_order if item not in seen),
                                    n_candidates))
    candidate_matrix = np.asarray([unseen_for_user(user) for user in active_users])
    if candidate_matrix.shape != (len(active_users), n_candidates):
        raise ValueError("Not enough unseen items for every user")
    keys = pd.DataFrame({"user_id": np.repeat(active_users, n_candidates),
                         "item_id": candidate_matrix.ravel()})
    keys = keys.merge(items, on="item_id", how="left")
    keys = keys.merge(users.rename(columns={"age": "user_age", "income": "user_income",
                                           "sex": "user_sex", "kids_flg": "user_kids"}),
                      on="user_id", how="left")
    keys["genre"] = keys.genre.fillna("unknown")
    positive = pd.MultiIndex.from_frame(future[["user_id", "item_id"]].drop_duplicates())
    y = pd.MultiIndex.from_frame(keys[["user_id", "item_id"]]).isin(positive).astype(np.uint8)
    features = pd.DataFrame(index=keys.index)
    for relation in RELATIONS:
        for days in WINDOWS:
            features[f"{relation}_{days}"] = lookup_counts(history, relation, keys, days, cutoff)
    for relation, key in [("user", "user_id"), ("item", "item_id")]:
        last_date = history.groupby(key).date.max()
        age = (cutoff - keys[key].map(last_date)).dt.days
        features[f"{relation}_recency"] = age.fillna(9999).to_numpy(dtype=np.float32)
    for days in [30, 180]:
        features[f"genre_share_{days}"] = features[f"genre_{days}"] / (features[f"user_{days}"] + 1)
    features["item_trend_7_30"] = features["item_7"] / (features["item_30"] + 1)
    for col in ["release_year", "age_rating", "for_kids", "user_kids"]:
        features[col] = pd.to_numeric(keys[col], errors="coerce").fillna(-1).astype(np.float32)
    for col in ["content_type", "genre_id", "user_age", "user_income", "user_sex"]:
        features[col] = pd.to_numeric(keys[col], errors="coerce").fillna(-1).astype(np.int32)
    features["user_kids"] = features["user_kids"].astype(np.int32)
    coverage = len(positive.intersection(pd.MultiIndex.from_frame(keys[["user_id", "item_id"]]))) / max(len(positive), 1)
    stats = {"cutoff": cutoff_string, "users": len(active_users), "candidates_per_user": n_candidates,
             "rows": len(keys), "positive_pairs": int(y.sum()), "future_pairs": len(positive),
             "candidate_coverage": coverage, "feature_seconds": round(time.monotonic() - t0, 2)}
    return features, y, stats, len(active_users)


def metrics(y, score, n_users, n_candidates):
    y = np.asarray(y).reshape(n_users, n_candidates)
    score = np.asarray(score).reshape(n_users, n_candidates)
    k = min(10, n_candidates)
    order = np.argsort(-score, axis=1)[:, :k]
    hits = np.take_along_axis(y, order, axis=1)
    weights = 1 / np.log2(np.arange(2, k + 2))
    dcg = (hits * weights).sum(axis=1)
    ideal = np.array([weights[:min(int(v), k)].sum() for v in y.sum(axis=1)])
    ndcg_per_user = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)
    ndcg = ndcg_per_user.mean()
    recall = np.divide(hits.sum(axis=1), y.sum(axis=1),
                       out=np.zeros(n_users), where=y.sum(axis=1) > 0).mean()
    flat_y, flat_score = y.ravel(), np.asarray(score).ravel()
    return {"ndcg@10": float(ndcg),
            "ndcg@10_given_candidate_positive": (
                float(ndcg_per_user[ideal > 0].mean()) if (ideal > 0).any() else None),
            "users_with_candidate_positive": int((ideal > 0).sum()),
            "recall@10": float(recall),
            "roc_auc": float(roc_auc_score(flat_y, flat_score)) if np.unique(flat_y).size == 2 else None,
            "logloss": float(log_loss(flat_y, np.clip(flat_score, 1e-7, 1-1e-7), labels=[0, 1]))}


def ndcg_by_user(y, score, n_users, n_candidates):
    y = np.asarray(y).reshape(n_users, n_candidates)
    score = np.asarray(score).reshape(n_users, n_candidates)
    k = min(10, n_candidates)
    hits = np.take_along_axis(y, np.argsort(-score, axis=1)[:, :k], axis=1)
    weights = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.array([weights[:min(int(v), k)].sum() for v in y.sum(axis=1)])
    return np.divide((hits * weights).sum(axis=1), ideal,
                     out=np.zeros(n_users), where=ideal > 0)


def train(x, y, cols):
    model = CatBoostClassifier(iterations=180, learning_rate=0.06, depth=6,
                               loss_function="Logloss", verbose=False, thread_count=4,
                               random_seed=42, l2_leaf_reg=5, allow_writing_files=False)
    model.fit(x[cols], y, cat_features=[c for c in CATEGORICAL if c in cols])
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("work/data_en.zip"))
    p.add_argument("--output", type=Path, default=Path("outputs/results.json"))
    p.add_argument("--users", type=int, default=3000)
    p.add_argument("--candidates", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    if args.users <= 0 or args.candidates < 10:
        p.error("--users must be positive and --candidates must be at least 10")
    t0 = time.monotonic()
    ev, items, users, pool_size = load_data(args.data, args.users, args.seed)
    snapshots = [make_snapshot(ev, items, users, c, args.candidates)
                 for c in TRAIN_CUTOFFS + [VAL_CUTOFF, TEST_CUTOFF]]
    train_x = pd.concat([x for x, _, _, _ in snapshots[:4]], ignore_index=True)
    train_y = np.concatenate([y for _, y, _, _ in snapshots[:4]])
    val_x, val_y, _, val_users = snapshots[4]
    test_x, test_y, _, test_users = snapshots[5]
    base_cols = MANUAL + STATIC
    generated_cols = [f"{r}_{w}" for r in RELATIONS for w in WINDOWS] + DERIVED + STATIC
    start = time.monotonic()
    base = train(train_x, train_y, base_cols)
    base_train_seconds = time.monotonic() - start
    start = time.monotonic()
    full = train(train_x, train_y, generated_cols)
    full_train_seconds = time.monotonic() - start
    gains = full.get_feature_importance()
    ranked = [generated_cols[i] for i in np.argsort(-gains)]
    choices = {}
    models = {len(generated_cols): full}
    training_seconds = {len(generated_cols): full_train_seconds}
    for k in [8, 16, len(generated_cols)]:
        if k not in models:
            start = time.monotonic()
            models[k] = train(train_x, train_y, ranked[:k])
            training_seconds[k] = time.monotonic() - start
        cols = generated_cols if k == len(generated_cols) else ranked[:k]
        pred = models[k].predict_proba(val_x[cols])[:, 1]
        choices[k] = metrics(val_y, pred, val_users, args.candidates)
    chosen_k = max(choices, key=lambda k: choices[k]["ndcg@10"])
    auto_cols = generated_cols if chosen_k == len(generated_cols) else ranked[:chosen_k]
    base_val = metrics(val_y, base.predict_proba(val_x[base_cols])[:, 1], val_users, args.candidates)
    base_test_pred = base.predict_proba(test_x[base_cols])[:, 1]
    auto_test_pred = models[chosen_k].predict_proba(test_x[auto_cols])[:, 1]
    base_test = metrics(test_y, base_test_pred, test_users, args.candidates)
    auto_test = metrics(test_y, auto_test_pred, test_users, args.candidates)
    pop_test = metrics(test_y, test_x["item_180"].to_numpy(), test_users, args.candidates)
    pop_test.pop("logloss")  # Counts are ranking scores, not calibrated probabilities.
    difference = ndcg_by_user(test_y, auto_test_pred, test_users, args.candidates) - ndcg_by_user(
        test_y, base_test_pred, test_users, args.candidates)
    rng = np.random.default_rng(args.seed + 1)
    bootstrap = np.array([difference[rng.integers(0, test_users, test_users)].mean()
                          for _ in range(1000)])
    result = {"configuration": vars(args) | {"data": str(args.data), "output": str(args.output)},
              "source_url": URL, "sampling_pool_users": pool_size,
              "sampled_events": len(ev), "snapshots": [s[2] for s in snapshots],
              "train_positive_pairs": int(train_y.sum()), "train_rows": len(train_y),
              "manual_features": base_cols, "generated_features": generated_cols,
              "gain_ranked_features": ranked, "validation_auto_choices": choices,
              "selected_k": chosen_k, "selected_auto_features": auto_cols,
              "metrics": {"popularity_test": pop_test, "manual_validation": base_val,
                          "manual_test": base_test, "auto_test": auto_test},
              "auto_minus_manual_ndcg_test": float(difference.mean()),
              "auto_minus_manual_ndcg_bootstrap_95pct": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
              "training_seconds": {"manual": base_train_seconds, "all_generated": full_train_seconds,
                                   "selected_auto": training_seconds[chosen_k]},
              "total_seconds": time.monotonic() - t0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": result["metrics"], "selected_k": chosen_k,
                      "total_seconds": result["total_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
