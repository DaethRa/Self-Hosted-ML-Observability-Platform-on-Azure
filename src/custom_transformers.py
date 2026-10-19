import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.features import haversine_distance

class OlistEnricher(BaseEstimator, TransformerMixin):
    def __init__(self, geo_agg, products_ref, sellers_ref, seller_te, cat_te, first_purchases_dict):
        self.geo_agg = geo_agg
        self.products_ref = products_ref
        self.sellers_ref = sellers_ref
        self.seller_te = seller_te
        self.cat_te = cat_te
        self.first_purchases_dict = first_purchases_dict

    def fit(self, X, y=None):
        return self

    def _aggregate_items(self, items_df, group_col):
        items = items_df.merge(self.products_ref, on='product_id', how='left')
        
        items['cat_bad_rate'] = self.cat_te.transform(items[['product_category_name']].astype(str))
        items['seller_bad_rate'] = self.seller_te.transform(items[['seller_id']])
        
        agg = items.groupby(group_col).agg(
            items_count=('order_item_id', 'count'),
            unique_products=('product_id', 'nunique'),
            unique_sellers=('seller_id', 'nunique'),
            total_price=('price', 'sum'),
            total_freight=('freight_value', 'sum'),
            total_weight_g=('product_weight_g', 'sum'),
            total_volume_cm3=('product_volume_cm3', 'sum'),
            avg_photos=('product_photos_qty', 'mean'),
            avg_desc_len=('product_description_lenght', 'mean'),
            max_seller_bad_rate=('seller_bad_rate', 'max'),
            max_cat_bad_rate=('cat_bad_rate', 'max'),
            first_seller_id=('seller_id', 'first')
        ).reset_index()
        
        agg = agg.merge(self.sellers_ref, left_on='first_seller_id', right_on='seller_id', how='left')
        return agg

    def transform(self, X):
        df = X.copy().reset_index(drop=True)
        
        items_records = []
        for idx, items in df['items_data'].items():
            if not isinstance(items, list) or len(items) == 0:
                raise ValueError(f"Row {idx} does not contain any items (items_data).")
            for item in items:
                item_copy = item.copy()
                item_copy['_row_idx'] = idx 
                items_records.append(item_copy)
                
        all_items = pd.DataFrame(items_records)
        agg_items = self._aggregate_items(all_items, group_col='_row_idx')

        df = df.merge(agg_items, left_index=True, right_on='_row_idx', how='left').drop(columns=['_row_idx'])
        df = df.drop(columns=['items_data'], errors='ignore')

        first_purchases = df['customer_unique_id'].map(self.first_purchases_dict)
        df['is_returning_customer'] = np.where(
            first_purchases.isna(), 0, 
            (df['order_purchase_timestamp'] > first_purchases).astype(int)
        )

        df['delay_days'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
        df['is_delayed'] = (df['delay_days'].fillna(0) > 0).astype(int)
        df['delay_days_positive'] = df['delay_days'].clip(lower=0)
        df['approval_delay'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds() / (3600 * 24)
        df['approval_delay'] = df['approval_delay'].fillna(0)
        df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
        df['is_weekend_purchase'] = df['purchase_dayofweek'].isin([5, 6]).astype(int)

        df['financial_diff'] = df['total_payment'] - df['total_price'] - df['total_freight']
        df['log_total_price'] = np.log1p(df['total_price'])
        df['freight_ratio'] = df['total_freight'] / (df['total_price'] + 1)
        df['is_split_order'] = (df['unique_sellers'] > 1).astype(int)
        df['avg_item_price'] = df['total_price'] / df['items_count'].clip(lower=1)

        df['log_weight'] = np.log1p(df['total_weight_g'])
        df['log_volume'] = np.log1p(df['total_volume_cm3'])
        df['product_density'] = df['total_weight_g'] / df['total_volume_cm3'].clip(lower=1)
        df['log_product_density'] = np.log1p(df['product_density'])

        df['is_cross_state'] = (df['customer_state'] != df['seller_state']).astype(int)

        df = df.merge(
            self.geo_agg.rename(columns={'geolocation_zip_code_prefix': 'customer_zip_code_prefix', 'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'}),
            on='customer_zip_code_prefix', how='left'
        )
        df = df.merge(
            self.geo_agg.rename(columns={'geolocation_zip_code_prefix': 'seller_zip_code_prefix', 'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}),
            on='seller_zip_code_prefix', how='left'
        )
        df['distance_km'] = haversine_distance(df['customer_lat'], df['customer_lng'], df['seller_lat'], df['seller_lng'])
        df['log_distance_km'] = np.log1p(df['distance_km'])

        critical_cols = ['total_price', 'total_weight_g', 'max_seller_bad_rate']
        if df[critical_cols].isna().any().any():
            raise ValueError("Enricher generated NaN values in critical features.")

        return df