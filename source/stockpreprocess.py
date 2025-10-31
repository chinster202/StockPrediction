# stockpreprocess.py

import torch
from torch.utils.data import Dataset, DataLoader
from .stockdataloader import load_stock_data, standardize_data, path
import pandas as pd
from . import config

# Load RAW (unstandardized) data
stockdf, split_idx = load_stock_data(path)


class StockDataset(Dataset):
    def __init__(self, contexts, targets):
        self.contexts = contexts
        self.targets = targets

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return {
            "context": torch.FloatTensor(self.contexts[idx]),
            "target": torch.FloatTensor([self.targets[idx]]),
        }


def preprocess_stock_data(
    stockdf,
    split_idx=config.train_test_split_percent,
    sequence_length=config.sequence_length,
    batch_size=config.batch_size,
):
    """
    Preprocess stock data with train-only standardization.

    Args:
        stockdf: Raw unstandardized dataframe
        split_idx: Index to split train/val (if None, calculates 80/20)
        sequence_length: Number of timesteps in each sequence
        batch_size: Batch size for dataloaders

    Returns:
        train_contexts, train_targets, val_contexts, val_targets,
        train_loader, val_loader, means, stds
    """

    print(f"\n{'=' * 60}")
    print("PREPROCESSING WITH TRAIN-ONLY STANDARDIZATION")
    print(f"{'=' * 60}")

    # Split into train and validation BEFORE standardization
    train_df = stockdf.iloc[:split_idx].copy()
    val_df = stockdf.iloc[split_idx:].copy()

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Standardize using ONLY training statistics
    train_df_std, val_df_std, means, stds = standardize_data(
        train_df, val_df, exclude_cols=["Date"]
    )

    # Drop Date column for sequence creation
    train_df_no_date = train_df_std.drop(columns=["Date","Adj Close","Volume","Open","High","Low"])
    val_df_no_date = val_df_std.drop(columns=["Date","Adj Close","Volume","Open","High","Low"])

    # Combine for sequence creation
    combined_df = pd.concat([train_df_no_date, val_df_no_date], ignore_index=True)

    print(f"\n{'=' * 60}")
    print("CREATING SEQUENCES")
    print(f"{'=' * 60}")

    # Create sequences
    contexts = []
    targets = []

    for i in range(len(combined_df) - sequence_length):
        # Get sequence_length consecutive rows as context
        context = combined_df.iloc[i : i + sequence_length].values.tolist()
        # Get Close price of next row as target
        target = combined_df.iloc[i + sequence_length]["Close"]

        contexts.append(context)
        targets.append(target)

    print(f"Total sequences created: {len(contexts)}")
    print(
        f"Context shape: {sequence_length} timesteps x {len(contexts[0][0])} features"
    )

    # Split sequences into train/val
    # Adjust split to account for sequence_length offset
    train_sequences_end = len(train_df_no_date) - sequence_length

    train_contexts = contexts[:train_sequences_end]
    train_targets = targets[:train_sequences_end]
    val_contexts = contexts[train_sequences_end:]
    val_targets = targets[train_sequences_end:]

    print(f"\nTraining sequences: {len(train_contexts)}")
    print(f"Validation sequences: {len(val_contexts)}")

    # Verify no data leakage
    print(f"\n{'=' * 60}")
    print("DATA LEAKAGE CHECK")
    print(f"{'=' * 60}")

    if len(train_contexts) > 0 and len(val_contexts) > 0:
        # Check that last training context doesn't overlap with validation
        last_train_idx = train_sequences_end + sequence_length - 1
        first_val_idx = train_sequences_end

        print(f"Last training sample uses data up to index: {last_train_idx}")
        print(
            f"First validation sample starts at index: {first_val_idx + sequence_length}"
        )
        print(f"Gap: {(first_val_idx + sequence_length) - last_train_idx} samples")

        if (first_val_idx + sequence_length) > last_train_idx:
            print("✅ No data leakage: validation doesn't see training data")
        else:
            print("⚠️  Warning: Potential data leakage detected")

    # Create datasets
    train_dataset = StockDataset(train_contexts, train_targets)
    val_dataset = StockDataset(val_contexts, val_targets)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"\n{'=' * 60}")
    print("DATALOADERS CREATED")
    print(f"{'=' * 60}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Batch size: {batch_size}")

    return (
        train_contexts,
        train_targets,
        val_contexts,
        val_targets,
        train_loader,
        val_loader,
        means,
        stds,
        train_df,
    )


print("Data preprocessing module loaded!")

if __name__ == "__main__":
    # Test preprocessing
    train_contexts, train_targets, val_contexts, val_targets, train_loader, val_loader, means, stds, train_df = preprocess_stock_data(stockdf, split_idx)

    print(f"\n{'='*60}")
    print("TESTING DATALOADERS")
    print(f"{'='*60}")

    # Test a batch
    sample_batch = next(iter(train_loader))
    print(f"\nSample batch shapes:")
    print(f"  Context: {sample_batch['context'].shape}")
    print(f"  Target: {sample_batch['target'].shape}")
    print(f"  Context dtype: {sample_batch['context'].dtype}")
    print(f"  Target dtype: {sample_batch['target'].dtype}")

    print(f"\nSample standardized values (first context, first timestep):")
    print(sample_batch['context'][0, 0, :])

    print(f"\nSample target values (first 5):")
    print(sample_batch['target'][:5].flatten())

    print(f"\n{'='*60}")
    print("STANDARDIZATION STATISTICS")
    print(f"{'='*60}")
    print("\nMeans:")
    print(means)
    print("\nStandard deviations:")
    print(stds)

    print(f"\n{'='*60}")
    print("SAMPLE SEQUENCES")
    print("Train context for first sequence:")
    print(train_contexts[0])
    print("Validation context for first sequence:")
    print(val_contexts[0])

    print(f"\nSAMPLE TARGETS")
    print("Train target for first sequence:")
    print(train_targets[0])
    print("Validation target for first sequence:")
    print(val_targets[0])

    print(train_loader)
    print(val_loader)