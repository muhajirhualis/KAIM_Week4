# src/proxy_target.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)

class HighRiskLabeler(BaseEstimator, TransformerMixin):
    """
    Engineer proxy target 'is_high_risk' using RFM + K-Means clustering.
    High-risk = cluster with lowest Frequency & Monetary (least engaged).
    """
    
    def __init__(self, snapshot_date=None, random_state=42):
        """
        Parameters
        ----------
        snapshot_date : Reference date for Recency calculation. 
            If None, uses max(TransactionStartTime).
        random_state : For reproducible clustering.
        """
        self.snapshot_date = pd.to_datetime(snapshot_date) if snapshot_date else None
        self.random_state = random_state
        self.scaler_ = None
        self.kmeans_ = None
        self.cluster_summary_ = None
        self.high_risk_cluster_ = None

    def _compute_rfm(self, df):
        """Compute RFM per CustomerId."""
        if 'CustomerId' not in df.columns:
            raise ValueError("Input DataFrame must contain 'CustomerId'")
        if 'TransactionStartTime' not in df.columns:
            raise ValueError("Input DataFrame must contain 'TransactionStartTime'")
        if 'Value' not in df.columns:
            raise ValueError("Input DataFrame must contain 'Value'")
        
        # Parse datetime (defensive)
        df = df.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        
        # Set snapshot date
        snapshot = self.snapshot_date or df['TransactionStartTime'].max()
        logger.info(f"Using snapshot date: {snapshot}")
        
        rfm = df.groupby('CustomerId').agg(
            Recency=('TransactionStartTime', lambda dates: (snapshot - dates.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Value', 'sum')
        ).reset_index()
        
        # Handle missing Recency (e.g., future dates)
        rfm['Recency'] = rfm['Recency'].clip(lower=0)
        return rfm
   
    def fit(self, X, y=None):
        """Fit RFM scaler and KMeans on training data."""
        rfm = self._compute_rfm(X)
        # Scale RFM (important for KMeans)
        self.scaler_ = StandardScaler()
        rfm_scaled = self.scaler_.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        
        # Cluster
        self.kmeans_ = KMeans(n_clusters=3, random_state=self.random_state, n_init=10)
        clusters = self.kmeans_.fit_predict(rfm_scaled)
        rfm['cluster'] = clusters
        
        # Identify high-risk cluster: lowest mean Frequency & Monetary
        summary = rfm.groupby('cluster')[['Frequency', 'Monetary']].mean()
        logger.info(f"Cluster RFM means:\n{summary}")
        
        # High-risk = min(Frequency + Monetary) or min(Frequency) 
        # (your EDA showed F & M highly correlated → use sum)
        summary['risk_score'] = summary['Frequency'] + summary['Monetary']
        self.high_risk_cluster_ = summary['risk_score'].idxmin()
        self.cluster_summary_ = summary
        
        logger.info(f"✅ High-risk cluster identified: {self.high_risk_cluster_}")
        return self
    
    def transform(self, X):
        """Add 'is_high_risk' column to input DataFrame (must have 'CustomerId')."""
        if self.kmeans_ is None:
            raise ValueError("fit() must be called before transform()")
        
        rfm = self._compute_rfm(X)
        rfm_scaled = self.scaler_.transform(rfm[['Recency', 'Frequency', 'Monetary']])
        clusters = self.kmeans_.predict(rfm_scaled)
        rfm['cluster'] = clusters
        rfm['is_high_risk'] = (rfm['cluster'] == self.high_risk_cluster_).astype(int)
        
        # Return only CustomerId + label (for clean merge)
        return rfm[['CustomerId', 'is_high_risk']]
    
