# src/eda_utils.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def eda_load_data(filepath: str) -> pd.DataFrame:
    """Load CSV with error handling."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {os.path.abspath(filepath)}")
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {e}")

      
def overview(df: pd.DataFrame):
    """Print shape, dtypes, memory usage."""
    print("=== Shape ===")
    print(f"{df.shape[0]:,} rows × {df.shape[1]} columns\n")
    print("=== Data Types ===")
    print(df.dtypes.value_counts())


def missing_and_duplicates(df: pd.DataFrame):
    """Report missing values and duplicates."""
    
    missing = df.isnull().sum()
    dup_rows = df.duplicated().sum()
    dup_ids = df['TransactionId'].duplicated().sum()
    print("=== Missing Values ===")
    if missing.sum() == 0:
        print("No missing values.")
    else:
        print(missing[missing > 0])
        
    print(f"\n=== Duplicates ===")
    print(f"Duplicate rows: {dup_rows}")
    print(f"Duplicate TransactionId: {dup_ids}")
    
def detect_outliers(df: pd.DataFrame, col: str, threshold_pct=0.99):
    """Detect and summarize extreme outliers above a percentile threshold."""
    threshold = df[col].quantile(threshold_pct)
    outliers = df[df[col] > threshold]
    n_out = len(outliers)
    pct_out = (n_out / len(df)) * 100
    print(f"\nOutliers in '{col}' > {threshold_pct*100:.1f}th pct ({threshold:,.0f}):")
    print(f"   → {n_out:,} records ({pct_out:.2f}%)")
    
    if n_out > 0:
        top_out = outliers.nlargest(5, col)
        print("   → Top 5:")
        display(top_out[['TransactionId', 'CustomerId', col, 'ProductCategory', 'CountryCode']])        
        
        # Boxplot
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        # Full range
        sns.boxplot(data=df, y=col, ax=ax[0])
        ax[0].set_title(f'{col} — Full Range')        
        # Capped at 99th percentile
        cap99 = df[col].quantile(0.99)
        sns.boxplot(data=df[df[col] <= cap99], y=col, ax=ax[1])
        ax[1].set_title(f'{col} — ≤99th Percentile ({cap99:,.0f})')
        
        plt.tight_layout()
        plt.show()
    
    return outliers

def plot_numeric(df: pd.DataFrame, col: str, log_scale=False):
    """Plot histogram (log optional) + capped boxplot."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    sns.histplot(df[col], bins=50, ax=ax[0], kde=False, log_scale=log_scale)
    ax[0].set_title(f'{col} Distribution')
    
    # Boxplot (99th percentile capped)
    cap = df[col].quantile(0.99)
    sns.boxplot(data=df[df[col] <= cap], y=col, ax=ax[1])
    ax[1].set_title(f'{col} (≤99th pct: {cap:,.0f})')
    
    plt.tight_layout()
    plt.show()

def plot_categorical(df: pd.DataFrame, col: str):

    """Plot top 10 categories."""
    top = df[col].value_counts().head(10)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=top.values, y=top.index.astype(str))
    plt.title(f'{col} — Top 10')
    plt.xlabel('Count')
    plt.show()

def correlation(df: pd.DataFrame):
    """Compute and plot correlation of user-level aggregates."""
    user_agg = (
        df.groupby('CustomerId')
        .agg(
            n_tx=('TransactionId', 'count'),
            total_value=('Value', 'sum'),
            avg_value=('Value', 'mean'),
            fraud_rate=('FraudResult', 'mean')
        )
        .reset_index()
    )
    corr = user_agg[['n_tx', 'total_value', 'avg_value', 'fraud_rate']].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("User-Level Correlations")
    plt.show()
    
    return user_agg