# train_arima.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import stockdataloader
from . import config
from .arima_model import StockARIMA, auto_arima_order


def plot_predictions(train_data, test_data, predictions, column='Close'):
    """Plot ARIMA predictions vs actual"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Full time series with predictions
    train_dates = range(len(train_data))
    test_dates = range(len(train_data), len(train_data) + len(test_data))
    
    ax1.plot(train_dates, train_data[column].values, label='Training Data', alpha=0.7)
    ax1.plot(test_dates, test_data[column].values, label='Actual (Test)', linewidth=2)
    ax1.plot(test_dates, predictions, label='Predicted', linewidth=2, linestyle='--')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Close Price ($)')
    ax1.set_title('ARIMA Stock Price Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoom in on test period
    ax2.plot(test_dates, test_data[column].values, label='Actual', marker='o', markersize=3)
    ax2.plot(test_dates, predictions, label='Predicted', marker='x', markersize=3)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Close Price ($)')
    ax2.set_title('ARIMA Predictions (Test Period Only)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/arima_predictions.png', dpi=150)
    plt.show()
    
    print(f"\n✅ Prediction plot saved to results/arima_predictions.png")


# def plot_residuals(model_fit):
#     """Plot residual diagnostics"""
    
#     fig = model_fit.plot_diagnostics(figsize=(15, 10))
#     plt.tight_layout()
#     plt.savefig('results/arima_residuals.png', dpi=150)
#     plt.close()
    
#     print(f"✅ Residual diagnostics saved to results/arima_residuals.png")


# def compare_predictions_sample(actuals, predictions, n_samples=10):
#     """Print sample predictions vs actuals"""
    
#     print(f"\n{'='*60}")
#     print(f"SAMPLE PREDICTIONS (first {n_samples})")
#     print(f"{'='*60}")
#     print(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Error %':<10}")
#     print(f"{'-'*60}")
    
#     for i in range(min(n_samples, len(actuals))):
#         error = predictions[i] - actuals[i]
#         error_pct = (error / actuals[i]) * 100
#         print(f"{i:<8} ${actuals[i]:<11.2f} ${predictions[i]:<11.2f} "
#               f"${error:<11.2f} {error_pct:<9.2f}%")

if __name__ == "__main__":
    print("="*60)
    print("ARIMA STOCK PRICE PREDICTION")
    print("="*60)
    
    # Load data (no standardization needed for ARIMA)
    stockdf, split_idx = stockdataloader.load_stock_data(config.path)
    
    # Split into train and test
    train_df = stockdf.iloc[:split_idx].copy()
    test_df = stockdf.iloc[split_idx:].copy()
    
    column = 'Close'
    
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
    order=(config.p, config.d, config.q),
    seasonal_order=seasonal_order
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
    #arima.plot_diagnostics(train_df, column=column, lags=40)
    
    # Option 1: Use auto ARIMA to find best order
    print(f"\n{'='*60}")
    print("FINDING OPTIMAL ARIMA ORDER")
    print(f"{'='*60}")
    user_choice = input("Use auto ARIMA to find best order? (y/n): ").lower()
    
    if user_choice == 'y':
        best_order = auto_arima_order(train_df, column=column, max_p=5, max_d=2, max_q=5)
        arima = StockARIMA(order=best_order,seasonal_order=seasonal_order)
    
    # Fit model
    arima.fit(train_df, column=column)
    
    # Plot residuals
    #plot_residuals(arima.model_fit)
    
    # Make rolling predictions on test set
    predictions, actuals = arima.predict_rolling(test_df, column=column)
    
    # Evaluate
    metrics = arima.evaluate(predictions, actuals)
    
    # Show sample predictions
    #compare_predictions_sample(actuals, predictions, n_samples=10)
    
    # Plot predictions
    plot_predictions(train_df, test_df, predictions, column=column)
    
    print("\n" + "="*60)
    print("ARIMA MODELING COMPLETE!")
    print("="*60)
    print(f"\nModel Order: {arima.order}")
    print(f"MAE: ${metrics['mae']:.2f}")
    print(f"RMSE: ${metrics['rmse']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%")
