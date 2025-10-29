
# import stockpreprocess
# import stockdataloader
# import source.config as config
from . import stockpreprocess
from . import stockdataloader
from . import config
from .model import StockLSTM, StockGRU
from . import train


def main():
    # Get preprocessed data
    # contexts_df, targets_df, means, stds = stockdataloader.load_stock_data(config.path)

    stockdf, split_idx = stockdataloader.load_stock_data(config.path)

    (
        train_context_data,
        train_target_data,
        val_context_data,
        val_target_data,
        train_loader,
        val_loader,
        means,  # From TRAINING data only
        stds,  # From TRAINING data only
        train_df,
    ) = stockpreprocess.preprocess_stock_data(stockdf, split_idx)

    # (
    #     _,
    #     _,
    #     _,
    #     _,
    #     train_loader,
    #     val_loader,
    # ) = stockpreprocess.preprocess_stock_data(contexts_df, targets_df)

    # Get input size from first batch
    # sample_batch = next(iter(train_loader))
    # seq_len = sample_batch['context'].shape[1]
    # input_size = sample_batch['context'].shape[2]
    # print(f"\nSequence length: {seq_len}, Input size (features): {input_size}")

    # Initialize model with config parameters
    if config.model_type == "StockGRU":
        model = StockGRU(
            input_size=config.input_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    else:
        model = StockLSTM(
            input_size=config.input_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

    print("\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Plot denormalized training contexts before training
    train.plot_train_contexts(train_df)

    # Train model
    print(
        f"\nStarting training for {config.epochs} epochs with learning rate {config.lr}"
    )
    train_losses, val_losses, val_target, val_output = train.train_model(
        model, train_loader, val_loader, epochs=config.epochs, lr=config.lr
    )

    print("")

    # Plot results
    train.plot_losses(train_losses, val_losses)

    # Plot predictions with denormalization
    train.plot_predictions(val_target, val_output, means, stds)

    print("\nTraining complete! Best model saved as 'best_stock_model.pth'")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Val Loss: {val_losses[-1]:.6f}")
    print(f"Best Val Loss: {min(val_losses):.6f}")


if __name__ == "__main__":
    main()
