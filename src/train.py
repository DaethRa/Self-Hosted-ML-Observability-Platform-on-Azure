import argparse
import json
import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
import joblib
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.features import build_geo, build_preprocessor, get_feature_lists
from src.custom_transformers import OlistEnricher
from src.data import data_loading_and_merging

API_PAYLOAD_COLS = [
    'order_id', 'customer_id', 'customer_unique_id', 'customer_zip_code_prefix', 'customer_state',
    'order_purchase_timestamp', 'order_approved_at', 'order_estimated_delivery_date',
    'order_delivered_customer_date', 'payment_count', 'total_payment', 'max_installments', 'items_data'
]

COST_PROMO = 10
COST_CHURN = 50
BEST_FN_WEIGHT = 3.0
BEST_FP_WEIGHT = 1.0

def parse_args():
    parser = argparse.ArgumentParser(description="Train Olist Bad Review Predictor")
    parser.add_argument('--data-dir', type=str, default='Datasets', help='Path to datasets')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save artifacts')
    parser.add_argument('--sample', type=float, default=1.0, help='Fraction of data to sample (for size limits)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def prepare_payload_format(master_df, order_items):
    df = master_df.copy()
    items_dict = order_items.groupby('order_id').apply(
        lambda x: x.to_dict(orient='records'), include_groups=False
    ).to_dict()
    df['items_data'] = df['order_id'].map(lambda x: items_dict.get(x, []))
    return df

def time_based_split(master_df, sample_frac=1.0):
    df = master_df.copy()
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        
    df = df[df['order_status'] == 'delivered'].copy()
    df = df.sort_values('review_creation_date')
    df = df.drop_duplicates(subset=['order_id'], keep='last')
    
    df = df.sort_values('order_purchase_timestamp')
    df['is_bad_review'] = df['review_score'].apply(lambda x: 1 if x in [1, 2] else 0)
    
    n = len(df)
    train_idx, val_idx = int(n * 0.6), int(n * 0.8)
    
    train_df = df.iloc[:train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[train_idx:val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[val_idx:].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df

def apply_kfold_te(train_df, all_items):
    train_df = train_df.copy()
    train_df['max_seller_bad_rate'] = 0.0
    train_df['max_cat_bad_rate'] = 0.0
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(train_df, train_df['is_bad_review']):
        fold_train_orders = train_df.iloc[train_idx][['order_id', 'is_bad_review']]
        fold_val_orders = train_df.iloc[val_idx][['order_id']] 

        fold_train_items = all_items.merge(fold_train_orders, on='order_id', how='inner')
        fold_val_items = all_items.merge(fold_val_orders, on='order_id', how='inner')

        seller_te = TargetEncoder(smoothing=10)
        cat_te = TargetEncoder(smoothing=10)
        
        seller_te.fit(fold_train_items[['seller_id']], fold_train_items['is_bad_review'])
        cat_te.fit(fold_train_items[['product_category_name']].astype(str), fold_train_items['is_bad_review'])

        fold_val_items['seller_bad_rate'] = seller_te.transform(fold_val_items[['seller_id']])
        fold_val_items['cat_bad_rate'] = cat_te.transform(fold_val_items[['product_category_name']].astype(str))

        val_risks = fold_val_items.groupby('order_id').agg(
            max_seller_bad_rate=('seller_bad_rate', 'max'),
            max_cat_bad_rate=('cat_bad_rate', 'max')
        ).reset_index()

        temp_merged = fold_val_orders.merge(val_risks, on='order_id', how='left')
        train_df.loc[val_idx, 'max_seller_bad_rate'] = temp_merged['max_seller_bad_rate'].values
        train_df.loc[val_idx, 'max_cat_bad_rate'] = temp_merged['max_cat_bad_rate'].values

    return train_df

def train():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    
    master_df, datasets = data_loading_and_merging(args.data_dir)
    
    products_ref = datasets['products'].copy()
    products_ref['product_volume_cm3'] = products_ref['product_length_cm'] * products_ref['product_height_cm'] * products_ref['product_width_cm']
    products_ref = products_ref[['product_id', 'product_category_name', 'product_weight_g', 'product_volume_cm3', 'product_photos_qty', 'product_description_lenght']]
    
    all_items = datasets['order_items'].merge(products_ref[['product_id', 'product_category_name']], on='product_id', how='left')
    master_df = prepare_payload_format(master_df, datasets['order_items'])
    
    train_df, val_df, test_df = time_based_split(master_df, args.sample)
    
    geo_agg = build_geo(datasets['geolocation'])
    sellers_ref = datasets['sellers'][['seller_id', 'seller_state', 'seller_zip_code_prefix']]
    
    first_purchases = train_df.groupby('customer_unique_id')['order_purchase_timestamp'].min().to_dict()
    
    train_df = apply_kfold_te(train_df, all_items)
    
    seller_te_full = TargetEncoder(smoothing=10)
    cat_te_full = TargetEncoder(smoothing=10)
    train_items_all = all_items.merge(train_df[['order_id', 'is_bad_review']], on='order_id', how='inner')
    seller_te_full.fit(train_items_all[['seller_id']], train_items_all['is_bad_review'])
    cat_te_full.fit(train_items_all[['product_category_name']].astype(str), train_items_all['is_bad_review'])

    enricher = OlistEnricher(
        geo_agg=geo_agg, products_ref=products_ref, sellers_ref=sellers_ref,
        seller_te=seller_te_full, cat_te=cat_te_full, first_purchases_dict=first_purchases
    )

    kfold_te_cols = train_df[['max_seller_bad_rate', 'max_cat_bad_rate']].copy()
    train_df_enriched = enricher.transform(train_df.drop(columns=['max_seller_bad_rate', 'max_cat_bad_rate']))
    
    assert (train_df_enriched['order_id'].values == train_df['order_id'].values).all(), "Index misalignment during enrichment"
    
    train_df_enriched['max_seller_bad_rate'] = kfold_te_cols['max_seller_bad_rate'].values
    train_df_enriched['max_cat_bad_rate'] = kfold_te_cols['max_cat_bad_rate'].values

    _, _, _, _, all_features = get_feature_lists()
    preprocessor = build_preprocessor()
    scale_weight = (len(train_df) - train_df['is_bad_review'].sum()) / train_df['is_bad_review'].sum()
    
    xgb_params = {
        'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 500,
        'subsample': 0.8, 'colsample_bytree': 0.6, 'reg_alpha': 0, 'reg_lambda': 5,
        'objective': 'binary:logistic', 'tree_method': 'hist', 'random_state': args.seed
    }
    
    tail_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(**xgb_params))
    ])

    weights = np.where(train_df_enriched['is_bad_review'] == 1, BEST_FN_WEIGHT, BEST_FP_WEIGHT)
    weights = np.where(train_df_enriched['review_score'] == 3, weights * 0.5, weights)
    
    tail_pipeline.fit(train_df_enriched, train_df_enriched['is_bad_review'], classifier__sample_weight=weights)

    production_pipeline = Pipeline([('enricher', enricher), ('tail', tail_pipeline)])

    X_val_payload = val_df[API_PAYLOAD_COLS]
    y_val = val_df['is_bad_review']
    y_val_proba = production_pipeline.predict_proba(X_val_payload)[:, 1]
    
    thresholds = np.linspace(0.01, 0.99, 200)
    val_costs = []
    
    for t in thresholds:
        y_pred_t = (y_val_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_t).ravel()
        val_costs.append((fp + tp) * COST_PROMO + fn * COST_CHURN)
        
    best_thresh = thresholds[np.argmin(val_costs)]

    X_test_payload = test_df[API_PAYLOAD_COLS]
    y_test = test_df['is_bad_review']
    y_test_proba = production_pipeline.predict_proba(X_test_payload)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_test_proba)
    pr_auc = average_precision_score(y_test, y_test_proba)
    
    y_test_pred = (y_test_proba >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    min_test_cost = (fp + tp) * COST_PROMO + fn * COST_CHURN
    _, _, fn_base, tp_base = confusion_matrix(y_test, np.zeros_like(y_test)).ravel()
    baseline_cost = (fn_base + tp_base) * COST_CHURN

    latencies = []
    _ = production_pipeline.predict_proba(X_test_payload.iloc[[0]]) 
    for i in range(50):
        start = time.perf_counter()
        _ = production_pipeline.predict_proba(X_test_payload.iloc[[i]])
        latencies.append((time.perf_counter() - start) * 1000)
    
    p50_ms = float(np.percentile(latencies, 50))
    p95_ms = float(np.percentile(latencies, 95))
    
    metadata = {
        "training_date": datetime.now().isoformat(),
        "git_sha": os.popen('git rev-parse HEAD').read().strip() if os.path.exists('.git') else "unknown",
        "seed": args.seed,
        "optimal_threshold": float(best_thresh),
        "metrics_test": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "savings": float(baseline_cost - min_test_cost)
        },
        "latency_ms": {
            "p50": p50_ms,
            "p95": p95_ms
        },
        "features": all_features,
        "versions": {
            "xgboost": xgb.__version__,
            "pandas": pd.__version__
        }
    }
    
    with open(os.path.join(args.model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    joblib.dump(production_pipeline, os.path.join(args.model_dir, 'production_pipeline.joblib'))
    
    val_reference_features = production_pipeline.named_steps['enricher'].transform(X_val_payload)
    val_reference_features[all_features].to_parquet(os.path.join(args.model_dir, 'reference.parquet'), index=False)
    
    test_df_sorted = test_df.sort_values('order_delivered_customer_date')
    replay_df = test_df_sorted[API_PAYLOAD_COLS]
    replay_df.to_json(os.path.join(args.model_dir, 'replay.jsonl'), orient='records', lines=True, date_format='iso')
    
    labels_df = test_df_sorted[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date', 'review_creation_date', 'review_score', 'is_bad_review']]
    labels_df.to_parquet(os.path.join(args.model_dir, 'labels.parquet'), index=False)
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("olist_bad_review_predictor")

    with mlflow.start_run() as run:
        mlflow.log_params(xgb_params)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("optimal_threshold", best_thresh)
        mlflow.log_param("features_count", len(all_features))

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("savings_dollars", baseline_cost - min_test_cost)
        mlflow.log_metric("latency_p50_ms", p50_ms)

        signature = infer_signature(
            X_test_payload.head(), 
            production_pipeline.predict_proba(X_test_payload.head())[:, 1]
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=production_pipeline,
            artifact_path="model",
            signature=signature,
            code_paths=["src/"], 
            registered_model_name="olist_model",
            serialization_format="cloudpickle" 
        )
        
        client = MlflowClient()
        client.set_registered_model_alias(
            name="olist_model", 
            alias="champion", 
            version=model_info.registered_model_version
        )
        
        mlflow.log_artifact(os.path.join(args.model_dir, 'metadata.json'))
        mlflow.log_artifact(os.path.join(args.model_dir, 'reference.parquet'))
        mlflow.log_artifact(os.path.join(args.model_dir, 'replay.jsonl'))
        mlflow.log_artifact(os.path.join(args.model_dir, 'labels.parquet'))

    print(f"Training completed. Threshold: {best_thresh:.2f}, Savings: ${baseline_cost - min_test_cost:,.0f}")
    print(f"Model saved to MLflow as 'olist_model' version {model_info.registered_model_version} with alias 'champion'")

if __name__ == "__main__":
    train()