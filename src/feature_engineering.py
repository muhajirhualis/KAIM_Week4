# src/data_processing.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

# Add WoE 
try:
    from xverse.transformer import WOE
    HAS_WOE = True
except ImportError:
    HAS_WOE = False
    logging.warning("xverse not installed — WoETransformer disabled.")

# 1. Aggregation Transformer (per CustomerId)
# ================================
class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Aggregate transaction-level data to customer-level."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert 'CustomerId' in X.columns, "Missing 'CustomerId'"
        
        # Aggregate — keep column names flat
        agg_dict = {
            'Amount': ['sum', 'mean', 'count'],
            'Value': ['sum', 'mean', 'std'],
            'FraudResult': 'mean',
            'TransactionStartTime': 'max'
        }
        agg_df = X.groupby('CustomerId').agg(agg_dict)
        
        # Flatten multi-index columns safely
        agg_df.columns = [
            '_'.join(col).strip().replace(' ', '_') 
            if isinstance(col, tuple) else col 
            for col in agg_df.columns
        ]
        agg_df = agg_df.reset_index()
        
        # Rename for clarity
        rename_map = {
            'Amount_sum': 'total_amount',
            'Amount_mean': 'avg_amount',
            'Amount_count': 'n_transactions',
            'Value_sum': 'total_value',
            'Value_mean': 'avg_value',
            'Value_std': 'std_value',
            'FraudResult_mean': 'fraud_rate',
            'TransactionStartTime_max': 'last_tx_time'
        }
        agg_df = agg_df.rename(columns=rename_map)
        
        # Handle std for single-transaction users
        agg_df['std_value'] = agg_df['std_value'].fillna(0)

        
        return agg_df
# ================================
# 2. Datetime Feature Extractor
# ================================
class DatetimeExtractor(BaseEstimator, TransformerMixin):
    """Extract hour, day, month, year from 'last_tx_time'."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        if 'last_tx_time' not in df.columns:
            return df  # skip if not aggregated yet
        dt = pd.to_datetime(df['last_tx_time'])
        df['tx_hour'] = dt.dt.hour
        df['tx_day'] = dt.dt.day
        df['tx_month'] = dt.dt.month
        df['tx_year'] = dt.dt.year
        return df.drop(columns=['last_tx_time'], errors='ignore')

# ================================
# 3. Categorical Encoder (One-Hot)
# ================================
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode top-N categories; drop rare ones."""
    
    def __init__(self, cat_cols=None, top_n=10):
        self.cat_cols = cat_cols or ['ProductCategory', 'ChannelId', 'CountryCode']
        self.top_n = top_n
        self.encoders_ = {}
        self.top_cats_ = {}
    
    def fit(self, X, y=None):
        for col in self.cat_cols:
            if col in X.columns:
                top_cats = X[col].value_counts().head(self.top_n).index.tolist()
                self.top_cats_[col] = top_cats
        return self
    
    def transform(self, X):
        df = X.copy()
        for col in self.cat_cols:
            if col in df.columns:
                # Replace rare with 'Other'
                df[col + '_enc'] = df[col].where(df[col].isin(self.top_cats_[col]), 'Other')
                # One-hot (no drop-first to preserve interpretability)
                ohe = pd.get_dummies(df[col + '_enc'], prefix=col, dtype=int)
                df = pd.concat([df, ohe], axis=1)
                df = df.drop(columns=[col, col + '_enc'])
        return df

# ================================
# 4. Missing Value Handler
# ================================
class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Impute missing numerics (mean/median) and categoricals (mode)."""
    
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imputers_ = {}
    
    def fit(self, X, y=None):
        for col in X.select_dtypes(include=[np.number]).columns:
            if self.strategy == 'mean':
                self.imputers_[col] = X[col].mean()
            elif self.strategy == 'median':
                self.imputers_[col] = X[col].median()
        return self
    
    def transform(self, X):
        df = X.copy()
        for col, val in self.imputers_.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        return df

# ================================
# 5. WoE Transformer (optional)
# ================================
class WoETransformer(BaseEstimator, TransformerMixin):
    """Apply Weight of Evidence to high-IV features (if xverse installed)."""
    
    def __init__(self, woe_cols=None):
        self.woe_cols = woe_cols or ['avg_value', 'n_transactions', 'fraud_rate']
        self.woe_models_ = {}
    
    def fit(self, X, y):
        if not HAS_WOE or y is None:
            return self
        for col in self.woe_cols:
            if col in X.columns:
                woe = WOE()
                try:
                    woe.fit(X[[col]], y)
                    self.woe_models_[col] = woe
                except Exception as e:
                    logging.warning(f"WoE failed for {col}: {e}")
        return self
    
    def transform(self, X):
        df = X.copy()
        if not HAS_WOE:
            return df
        for col, woe in self.woe_models_.items():
            if col in df.columns:
                df[col + '_woe'] = woe.transform(df[[col]]).iloc[:, 0]
        return df

# ================================
# 6. Full Pipeline Builder
# ================================

    def build_feature_pipeline(use_woe=False):
        """Build pipeline that preserves CustomerId and outputs clean DataFrame."""
        # Step 1: Aggregate → adds CustomerId + numeric RFM features
        agg = CustomerAggregator()
        
        # Step 2: Datetime extraction
        dt = DatetimeExtractor()
        
        # Step 3: Categorical encoding
        cat = CategoricalEncoder()
        
        # Step 4: Missing value handler
        miss = MissingValueHandler(strategy='median')
        
        # Step 5: Scaling — ONLY numeric columns (exclude CustomerId, one-hot cols)
        numeric_features = [
            'total_amount', 'avg_amount', 'n_transactions',
            'total_value', 'avg_value', 'std_value', 'fraud_rate',
            'tx_hour', 'tx_day', 'tx_month', 'tx_year'
        ]
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
            ],
            remainder='passthrough',  # keeps CustomerId & one-hot features
            verbose_feature_names_out=False
        )
        
        steps = [
            ('aggregator', agg),
            ('datetime', dt),
            ('categorical', cat),
            ('missing', miss),
            ('scaler', preprocessor)
        ]
        
        if use_woe:
            steps.append(('woe', WoETransformer()))
        
        return Pipeline(steps)