import pandas as pd
import numpy as np
from . import config

path = config.path


def load_stock_data(path=path):
    stockdf = pd.read_csv(path)
    print(f"Total examples in dataset: {len(stockdf)}")

    # Store original statistics
    means = {}
    stds = {}

    # Normalize all columns except Date
    cols_to_standardize = stockdf.columns.difference(["Date"])

    print(cols_to_standardize)

    # Handle Volume separately if it's causing issues
    volume_col = 'Volume' if 'Volume' in cols_to_standardize else None
    
    # Standardize price columns
    price_cols = [col for col in cols_to_standardize if col != 'Volume']

    print(price_cols)

    print(volume_col)

    for col in price_cols:
        means[col] = stockdf[col].mean()
        stds[col] = stockdf[col].std()
        stockdf[col] = (stockdf[col] - means[col]) / stds[col]
    
    # Handle Volume with log transform (optional but recommended)
    if volume_col:
        # Log transform volume to reduce scale
        stockdf[volume_col] = np.log1p(stockdf[volume_col])
        means[volume_col] = stockdf[volume_col].mean()
        stds[volume_col] = stockdf[volume_col].std()
        stockdf[volume_col] = (stockdf[volume_col] - means[volume_col]) / stds[volume_col]
    
    # Store mean and std BEFORE standardization
    # means = stockdf[cols_to_standardize].mean()
    # stds = stockdf[cols_to_standardize].std()
    
    # # Standardize
    # stockdf[cols_to_standardize] = (
    #     stockdf[cols_to_standardize] - means
    # ) / stds

    contexts_df = stockdf.iloc[0 : (len(stockdf) - 1)]
    targets_df = stockdf.iloc[7:]

    return contexts_df, targets_df, means, stds

# stockdataloader.py

def denormalize_predictions(standardized_predictions, column_name='Close', means=None, stds=None):

    mean = means[column_name]
    std = stds[column_name]

    unstandardized_column = (standardized_predictions * std) + mean
    
    # Reverse standardization: original = (standardized * std) + mean
    return unstandardized_column


if __name__ == "__main__":
    contexts_df, targets_df, means, stds = load_stock_data(path)

    print(means)
    print(stds)

    print(f"Training examples: {len(contexts_df)}")
    print(f"Validation examples: {len(targets_df)}")

    print(contexts_df[:3])
    print(targets_df[:3])

    print(contexts_df.head())
    print(f'contexts_df.tail()\n')

    print((contexts_df.describe()))
    print((targets_df.describe()))
