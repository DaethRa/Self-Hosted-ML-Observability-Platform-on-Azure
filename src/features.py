import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0
    return c * r

def build_geo(geo_df):
    return geo_df.groupby('geolocation_zip_code_prefix')[
        ['geolocation_lat', 'geolocation_lng']
    ].mean().reset_index()

def get_feature_lists():
    mean_features = ['max_seller_bad_rate', 'max_cat_bad_rate', 'freight_ratio', 'approval_delay', 'avg_photos', 'avg_desc_len'] 
    median_features = ['delay_days_positive', 'log_total_price', 'log_weight', 'log_volume', 'total_freight', 'max_installments', 'avg_item_price', 'financial_diff', 'log_product_density', 'log_distance_km'] 
    one_features = ['items_count', 'unique_products', 'unique_sellers', 'payment_count', 'is_delayed', 'is_split_order', 'is_weekend_purchase', 'is_cross_state', 'is_returning_customer']
    cat_features = ['customer_state'] 
    
    all_features = mean_features + median_features + one_features + cat_features
    return mean_features, median_features, one_features, cat_features, all_features

def build_preprocessor():
    mean_features, median_features, one_features, cat_features, _ = get_feature_lists()
    
    mean_pipe = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
    median_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    freq_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])
    
    return ColumnTransformer(
        transformers=[
            ('mean_num', mean_pipe, mean_features),
            ('median_num', median_pipe, median_features),
            ('freq_num', freq_pipe, one_features),
            ('cat_enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ],
        remainder='drop'
    )