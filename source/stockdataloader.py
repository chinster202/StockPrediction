# import pandas as pd
# # import config

# path = "data/A.csv" # "../data/A.csv"

# def load_stock_data(path=path):
#     stockdf = pd.read_csv(path)
#     print(f"Total examples in dataset: {len(stockdf)}")

#     # Normalize all columns except Date
#     cols_to_standardize = stockdf.columns.difference(["Date"])

#     print(cols_to_standardize)
#     # stockdf[cols_to_normalize] = (stockdf[cols_to_normalize] - stockdf[cols_to_normalize].min()) / (stockdf[cols_to_normalize].max() - stockdf[cols_to_normalize].min())
#     stockdf[cols_to_standardize] = (
#         stockdf[cols_to_standardize] - stockdf[cols_to_standardize].mean()
#     ) / stockdf[cols_to_standardize].std()

#     contexts_df = stockdf.iloc[0 : (len(stockdf) - 1)]
#     targets_df = stockdf.iloc[7:]

#     # print(f"Training examples: {len(contexts_df)}")
#     # print(f"Validation examples: {len(val_df)}")

#     # assert len(stockdf) - 1 == len(contexts_df)
#     # assert len(stockdf) - len(targets_df) == 7

#     return contexts_df, targets_df

# stockdataloader.py

import pandas as pd

path = "data/A.csv"


def load_stock_data(path=path):
    stockdf = pd.read_csv(path)
    print(f"Total examples in dataset: {len(stockdf)}")

    # Normalize all columns except Date
    cols_to_standardize = stockdf.columns.difference(["Date"])

    print(cols_to_standardize)
    
    # Store mean and std BEFORE standardization
    means = stockdf[cols_to_standardize].mean()
    stds = stockdf[cols_to_standardize].std()
    
    # Standardize
    stockdf[cols_to_standardize] = (
        stockdf[cols_to_standardize] - means
    ) / stds

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


# if __name__ == "__main__":
#     contexts_df, targets_df, means, stds = load_stock_data(path)

#     print(means)
#     print(stds)

#     print(f"Training examples: {len(contexts_df)}")
#     print(f"Validation examples: {len(targets_df)}")

#     print(contexts_df[:3])
#     print(targets_df[:3])

#     print((contexts_df.head()))
