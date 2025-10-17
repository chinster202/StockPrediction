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


# Set device and enable cuDNN benchmarking for faster training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

stockdf = pd.read_csv("../data/A.csv")
print(f"Total examples in dataset: {len(stockdf)}")

train_df = stockdf.iloc[0:(len(stockdf)-1)]
val_df = stockdf.iloc[7:]

print(f"Training examples: {len(train_df)}")
print(f"Validation examples: {len(val_df)}")

assert len(stockdf) - 1 == len(train_df)
assert len(stockdf) - len(val_df) == 7

print((train_df.head()))