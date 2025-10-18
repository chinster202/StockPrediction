import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
import json
from tqdm.auto import tqdm  # for progress bars
import config


# Set device and enable cuDNN benchmarking for faster training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

path = "../data/A.csv"

def load_stock_data(path=path):

    stockdf = pd.read_csv(path)
    print(f"Total examples in dataset: {len(stockdf)}")

    # Normalize all columns except Date
    cols_to_standardize = stockdf.columns.difference(['Date'])

    print(cols_to_standardize)
    #stockdf[cols_to_normalize] = (stockdf[cols_to_normalize] - stockdf[cols_to_normalize].min()) / (stockdf[cols_to_normalize].max() - stockdf[cols_to_normalize].min())
    stockdf[cols_to_standardize] = (stockdf[cols_to_standardize] - stockdf[cols_to_standardize].mean()) / stockdf[cols_to_standardize].std()
    
    contexts_df = stockdf.iloc[0:(len(stockdf)-1)]
    targets_df = stockdf.iloc[7:]

    #print(f"Training examples: {len(contexts_df)}")
    #print(f"Validation examples: {len(val_df)}")

    assert len(stockdf) - 1 == len(contexts_df)
    assert len(stockdf) - len(targets_df) == 7

    return contexts_df, targets_df

if __name__ == "__main__":
    contexts_df, targets_df = load_stock_data(path)
    
    print(f"Training examples: {len(contexts_df)}")
    print(f"Validation examples: {len(targets_df)}")

    print(contexts_df[:3])
    print(targets_df[:3])

    print((contexts_df.head()))

