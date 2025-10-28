# stockdataloader.py

import pandas as pd
import numpy as np
from . import config

path = "data/A.csv"


def load_stock_data(path=config.path, split_ratio=config.train_test_split_percent):
    """
    Load and prepare stock data WITHOUT standardization.
    Standardization will be done in preprocessing using only training data.
    
    Args:
        path: Path to CSV file
        split_ratio: Train/test split ratio (used to know where split will occur)
    
    Returns:
        stockdf: Raw (unstandardized) dataframe
        split_idx: Index where train/val split occurs
    """
    stockdf = pd.read_csv(path)
    print(f"Total examples in dataset: {len(stockdf)}")
    
    # Check for missing values
    if stockdf.isnull().any().any():
        print("⚠️  Warning: Dataset contains missing values. Filling with forward fill.")
        stockdf = stockdf.fillna(method='ffill').fillna(method='bfill')
    
    # Calculate split index
    split_idx = int(len(stockdf) * split_ratio)
    
    print(f"\nData will be split at index {split_idx}")
    print(f"Training samples: {split_idx}")
    print(f"Validation samples: {len(stockdf) - split_idx}")
    
    # Show price ranges (before standardization)
    print(f"\nPrice ranges in raw data:")
    print(f"Full dataset Close: ${stockdf['Close'].min():.2f} - ${stockdf['Close'].max():.2f}")
    print(f"Training Close: ${stockdf.iloc[:split_idx]['Close'].min():.2f} - ${stockdf.iloc[:split_idx]['Close'].max():.2f}")
    print(f"Validation Close: ${stockdf.iloc[split_idx:]['Close'].min():.2f} - ${stockdf.iloc[split_idx:]['Close'].max():.2f}")
    
    return stockdf, split_idx


def standardize_data(train_df, val_df, exclude_cols=['Date']):
    """
    Standardize data using ONLY training statistics.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        exclude_cols: Columns to exclude from standardization
    
    Returns:
        train_df_std: Standardized training data
        val_df_std: Standardized validation data
        means: Mean values from training data
        stds: Std values from training data
    """
    # Get columns to standardize
    cols_to_standardize = train_df.columns.difference(exclude_cols)
    
    # Calculate statistics ONLY from training data
    means = train_df[cols_to_standardize].mean()
    stds = train_df[cols_to_standardize].std()
    
    print(f"\n{'='*60}")
    print("STANDARDIZATION STATISTICS (from training data only)")
    print(f"{'='*60}")
    
    for col in cols_to_standardize:
        print(f"{col:12s}: mean={means[col]:10.2f}, std={stds[col]:10.2f}")
    
    # Standardize both train and validation using TRAINING statistics
    train_df_std = train_df.copy()
    val_df_std = val_df.copy()
    
    train_df_std[cols_to_standardize] = (train_df[cols_to_standardize] - means) / stds
    val_df_std[cols_to_standardize] = (val_df[cols_to_standardize] - means) / stds
    
    # Check for extreme values in validation set
    print(f"\n{'='*60}")
    print("CHECKING FOR DISTRIBUTION SHIFT")
    print(f"{'='*60}")
    
    for col in cols_to_standardize:
        train_range = (train_df_std[col].min(), train_df_std[col].max())
        val_range = (val_df_std[col].min(), val_df_std[col].max())
        
        print(f"{col:12s}: train range [{train_range[0]:6.2f}, {train_range[1]:6.2f}], "
              f"val range [{val_range[0]:6.2f}, {val_range[1]:6.2f}]")
        
        # Warn if validation data goes beyond training range
        if val_range[0] < train_range[0] - 1 or val_range[1] > train_range[1] + 1:
            print(f"  ⚠️  Warning: Validation data for {col} extends beyond training range")
    
    return train_df_std, val_df_std, means, stds


def denormalize_predictions(predictions, column_name='Close', means=None, stds=None):
    """
    Denormalize standardized predictions back to original scale.
    
    Args:
        predictions: Standardized predictions (numpy array or tensor)
        column_name: Name of the column (e.g., 'Close')
        means: Series or dict of mean values
        stds: Series or dict of std values
    
    Returns:
        Denormalized predictions in original scale
    """
    if means is None or stds is None:
        raise ValueError("Must provide means and stds for denormalization")
    
    mean = means[column_name]
    std = stds[column_name]
    
    # Reverse standardization: original = (standardized * std) + mean
    return (predictions * std) + mean


# if __name__ == "__main__":
#     # Test loading
#     stockdf, split_idx = load_stock_data(path)
    
#     print(f"\nLoaded dataframe shape: {stockdf.shape}")
#     print(f"\nFirst few rows:")
#     print(stockdf.head())
#     print(f"\nLast few rows:")
#     print(stockdf.tail())
    
#     # Test standardization
#     train_df = stockdf.iloc[:split_idx]
#     val_df = stockdf.iloc[split_idx:]
    
#     train_df_std, val_df_std, means, stds = standardize_data(train_df, val_df)
    
#     print(f"\n{'='*60}")
#     print("STANDARDIZED DATA SAMPLE")
#     print(f"{'='*60}")
#     print("\nTraining data (first 3 rows):")
#     print(train_df_std.head(3))
#     print("\nValidation data (first 3 rows):")
#     print(val_df_std.head(3))

# import pandas as pd
# import numpy as np
# from . import config
# import os

# def load_stock_data(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Data file not found: {path}")
#     stockdf = pd.read_csv(path)
#     print(f"Total examples in dataset: {len(stockdf)}")

#     # Store original statistics
#     means = {}
#     stds = {}

#     # Normalize all columns except Date
#     cols_to_standardize = stockdf.columns.difference(["Date"])

#     print(cols_to_standardize)

#     # Handle Volume separately if it's causing issues
#     volume_col = 'Volume' if 'Volume' in cols_to_standardize else None
    
#     # Standardize price columns
#     price_cols = [col for col in cols_to_standardize if col != 'Volume']

#     print(price_cols)

#     print(volume_col)

#     for col in price_cols:
#         means[col] = stockdf[col].mean()
#         stds[col] = stockdf[col].std()
#         stockdf[col] = (stockdf[col] - means[col]) / stds[col]
    
#     # Handle Volume with log transform (optional but recommended)
#     if volume_col:
#         # Log transform volume to reduce scale
#         stockdf[volume_col] = np.log1p(stockdf[volume_col])
#         means[volume_col] = stockdf[volume_col].mean()
#         stds[volume_col] = stockdf[volume_col].std()
#         stockdf[volume_col] = (stockdf[volume_col] - means[volume_col]) / stds[volume_col]
    
#     # Store mean and std BEFORE standardization
#     # means = stockdf[cols_to_standardize].mean()
#     # stds = stockdf[cols_to_standardize].std()
    
#     # # Standardize
#     # stockdf[cols_to_standardize] = (
#     #     stockdf[cols_to_standardize] - means
#     # ) / stds

#     contexts_df = stockdf.iloc[0 : (len(stockdf) - 1)]
#     targets_df = stockdf.iloc[14:]

#     return contexts_df, targets_df, means, stds

# # stockdataloader.py

# def denormalize_predictions(standardized_predictions, column_name='Close', means=None, stds=None):

#     mean = means[column_name]
#     std = stds[column_name]

#     unstandardized_column = (standardized_predictions * std) + mean
    
#     # Reverse standardization: original = (standardized * std) + mean
#     return unstandardized_column


# if __name__ == "__main__":
#     contexts_df, targets_df, means, stds = load_stock_data(config.path)

#     print(means)
#     print(stds)

#     print(f"Training examples: {len(contexts_df)}")
#     print(f"Validation examples: {len(targets_df)}")

#     print(contexts_df[:3])
#     print(targets_df[:3])

#     print(contexts_df.head())
#     print(f'contexts_df.tail()\n')

#     print((contexts_df.describe()))
#     print((targets_df.describe()))
