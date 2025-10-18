from numpy import floor
import stockdataloader
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import config

contexts_df, targets_df = stockdataloader.load_stock_data(stockdataloader.path)

class StockDataset(Dataset):
    def __init__(self, contexts, targets):
        self.contexts = contexts
        self.targets = targets
    
    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        return {
            'context': self.contexts[idx],
            'target': self.targets[idx]
        }

def preprocess_stock_data(contexts_df, targets_df):

    contexts_df_no_date = contexts_df.drop(columns=['Date'])

    contexts = []
    targets = []
    for i in range(len(targets_df) - 1):
        contexts.append(contexts_df_no_date.iloc[i:i+6].values.tolist())
        targets.append(targets_df.iloc[i]['Close'])

    # Train test split
    train_context_data = contexts[:int(len(contexts)*config.train_test_split_percent)]
    train_target_data = targets[:int(len(targets)*config.train_test_split_percent)]
    val_context_data = contexts[int(len(contexts)*config.train_test_split_percent):]
    val_target_data = targets[int(len(targets)*config.train_test_split_percent):]

    # Convert to tensors
    train_context_data = torch.tensor(train_context_data, dtype=torch.long)
    train_target_data = torch.tensor(train_target_data, dtype=torch.long)
    val_context_data = torch.tensor(val_context_data, dtype=torch.long)
    val_target_data = torch.tensor(val_target_data, dtype=torch.long)

    # Create datasets
    train_dataset = StockDataset(
        train_context_data,
        train_target_data
    )

    val_dataset = StockDataset(
        val_context_data,
        val_target_data
    )

    # Create data loaders with larger batch size and pin_memory for faster data transfer
    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        #shuffle=True,
       #num_workers=2 if torch.cuda.is_available() else 0,
       #pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        #shuffle=False,
        #num_workers=2 if torch.cuda.is_available() else 0,
        #pin_memory=torch.cuda.is_available()
    )
    
    return train_context_data, train_target_data, val_context_data, val_target_data, train_loader, val_loader

#train_context_data, train_target_data, val_context_data, val_target_data = preprocess_stock_data(contexts_df, targets_df)

print("Data preprocessing completed!")

if __name__ == "__main__":
    train_context_data, train_target_data, val_context_data, val_target_data, train_loader, val_loader = preprocess_stock_data(contexts_df, targets_df)
    print(f"Number of training contexts: {len(train_context_data)}")
    print(f"Number of training targets: {len(train_target_data)}")
    print("First context:", train_context_data[0])
    print("First target:", train_target_data[0])

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")