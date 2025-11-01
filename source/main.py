# import stockpreprocess
# import stockdataloader
# import source.config as config
from . import stockpreprocess
from . import stockdataloader
from . import config
from .model import StockLSTM, StockGRU
from . import train
from .arima_model import StockARIMA, auto_arima_order, plot_predictions, plot_residuals
from .trading_strategy import TradingSimulator, run_parameter_sweep


def main():
    # Get preprocessed data
    stockdf, split_idx = stockdataloader.load_stock_data(config.path)

    if config.model == "RNN":
        (
            _,
            _,
            _,
            _,
            train_loader,
            val_loader,
            means,  # From TRAINING data only
            stds,  # From TRAINING data only
            train_df,
        ) = stockpreprocess.preprocess_stock_data(stockdf, split_idx)

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

    else:
        print("=" * 60)
        print("ARIMA STOCK PRICE PREDICTION")
        print("=" * 60)

        # Split into train and test
        train_df = stockdf.iloc[:split_idx].copy()
        test_df = stockdf.iloc[split_idx:].copy()

        column = "Close"

        print(f"\nPredicting column: {column}")
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")

        # Determine seasonal order based on config
        if config.arima_type == "SARIMA":
            seasonal_order = (config.P, config.D, config.Q, config.s)
        else:
            seasonal_order = None

        # Initialize ARIMA model
        arima = StockARIMA(
            order=(config.p, config.d, config.q), seasonal_order=seasonal_order
        )
        # (p, d, q)
        # seasonal_order=(0, 1, 1, 21)  # Only seasonal MA component with seasonality of 21 (trading days in a month)
        # seasonal_order=(2, 1, 2, 21)  # Both seasonal AR and MA components with seasonality of 21

        # You can modify the order and seasonal_order as needed.
        # p = AR order (autoregressive)
        # d = differencing order
        # q = MA order (moving average)
        # P = Seasonal AR order
        # D = Seasonal differencing order
        # Q = Seasonal MA order
        # s = Seasonal period (e.g., 12 for monthly data, 252 for daily stock data)

        # Check stationarity
        arima.check_stationarity(train_df, column=column)

        # Plot diagnostics to help choose p and q
        # arima.plot_diagnostics(train_df, column=column, lags=40)

        # Option 1: Use auto ARIMA to find best order
        print(f"\n{'=' * 60}")
        print("FINDING OPTIMAL ARIMA ORDER")
        print(f"{'=' * 60}")
        user_choice = input("Use auto ARIMA to find best order? (y/n): ").lower()

        if user_choice == "y":
            best_order = auto_arima_order(
                train_df, column=column, max_p=5, max_d=2, max_q=5
            )
            arima = StockARIMA(order=best_order, seasonal_order=seasonal_order)

        # Fit model
        arima.fit(train_df, column=column)

        # Plot residuals
        # plot_residuals(arima.model_fit)

        # Make rolling predictions on test set
        predictions, actuals = arima.predict_rolling(test_df, column=column)

        # Evaluate
        metrics = arima.evaluate(predictions, actuals)

        # Show sample predictions
        # compare_predictions_sample(actuals, predictions, n_samples=10)

        # Plot predictions
        plot_predictions(train_df, test_df, predictions, column=column)

        # Plot residuals
        plot_residuals(test_df, predictions, column=column)

        print("\n" + "=" * 60)
        print("ARIMA MODELING COMPLETE!")
        print("=" * 60)
        print(f"\nModel Order: {arima.order}")
        print(f"MAE: ${metrics['mae']:.2f}")
        print(f"RMSE: ${metrics['rmse']:.2f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%")

        # Run trading simulation
        print("\n" + "=" * 80)
        print("TRADING SIMULATION")
        print("=" * 80)

        simulator = TradingSimulator(
            initial_capital=100000,
            n_days_threshold=9,
            transaction_fee=0.001,  # 0.1% transaction fee
        )

        trading_results = simulator.simulate(actuals, predictions)
        simulator.plot_results(actuals)

        # Optional: Run parameter sweep to find best n_days_threshold
        print("\n" + "=" * 80)
        run_sweep = input(
            "Run parameter sweep to find optimal n_days_threshold? (y/n): "
        ).lower()

        if run_sweep == "y":
            sweep_results = run_parameter_sweep(
                actuals, predictions, n_days_range=range(1, 11), initial_capital=100000
            )


if __name__ == "__main__":
    main()
