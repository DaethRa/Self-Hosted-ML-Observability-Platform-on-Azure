import pandas as pd
from pathlib import Path

def data_loading_and_merging(data_dir: str | Path = "Datasets"):
    data_path = Path(data_dir)
    
    customers = pd.read_csv(data_path / "olist_customers_dataset.csv")
    order_items = pd.read_csv(data_path / "olist_order_items_dataset.csv")
    payments = pd.read_csv(data_path / "olist_order_payments_dataset.csv")
    reviews = pd.read_csv(data_path / "olist_order_reviews_dataset.csv")
    orders = pd.read_csv(data_path / "olist_orders_dataset.csv")
    geolocation = pd.read_csv(data_path / "olist_geolocation_dataset.csv")
    products = pd.read_csv(data_path / "olist_products_dataset.csv")
    sellers = pd.read_csv(data_path / "olist_sellers_dataset.csv")

    datasets = {
        "customers": customers, "geolocation": geolocation, "order_items": order_items,
        "payments": payments, "reviews": reviews, "orders": orders,
        "products": products, "sellers": sellers,
    }

    payments_agg = payments.groupby("order_id").agg(
        payment_count=("payment_type", "count"),
        total_payment=("payment_value", "sum"),
        max_installments=("payment_installments", "max")
    ).reset_index()

    reviews_subset = reviews[['order_id', 'review_score', 'review_creation_date']]

    master_df = (
        orders.merge(payments_agg, on="order_id", how="left")
        .merge(customers[['customer_id', 'customer_unique_id', 'customer_zip_code_prefix', 'customer_state']], on="customer_id", how="left")
        .merge(reviews_subset, on="order_id", how="inner")
    )

    date_cols = ['order_purchase_timestamp', 'order_estimated_delivery_date', 
                 'order_delivered_customer_date', 'order_approved_at', 'review_creation_date']

    for col in date_cols:
        master_df[col] = pd.to_datetime(master_df[col])

    return master_df, datasets