#arima_model.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class StockARIMA:
    """ARIMA model for stock price prediction"""
    
    def __init__(self, order=(5, 1, 0), seasonal_order=None):
        """
        Initialize ARIMA model
        
        Args:
            order: (p, d, q) - (AR order, differencing order, MA order)
            seasonal_order: (P, D, Q, s) for SARIMA, None for regular ARIMA
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.model_fit = None
        self.is_fitted = False
        
    def check_stationarity(self, data, column='Close'):
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        
        Args:
            data: DataFrame with time series data
            column: Column to test
        
        Returns:
            Dictionary with test results
        """
        series = data[column].dropna()
        
        result = adfuller(series)
        
        print(f"\n{'='*60}")
        print(f"STATIONARITY TEST: {column}")
        print(f"{'='*60}")
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"P-value: {result[1]:.6f}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.3f}")
        
        is_stationary = result[1] < 0.05
        
        if is_stationary:
            print(f"✅ Series is STATIONARY (p-value < 0.05)")
        else:
            print(f"⚠️  Series is NOT STATIONARY (p-value >= 0.05)")
            print(f"   Consider differencing or increasing d parameter")
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': is_stationary
        }
    
    # def plot_diagnostics(self, data, column='Close', lags=40):
    #     """Plot ACF and PACF to help determine p and q parameters"""
        
    #     series = data[column].dropna()
        
    #     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
    #     # Original series
    #     axes[0, 0].plot(series)
    #     axes[0, 0].set_title(f'Original {column} Series')
    #     axes[0, 0].set_xlabel('Time')
    #     axes[0, 0].set_ylabel('Price')
    #     axes[0, 0].grid(True, alpha=0.3)
        
    #     # Differenced series
    #     diff_series = series.diff().dropna()
    #     axes[0, 1].plot(diff_series)
    #     axes[0, 1].set_title(f'First Difference of {column}')
    #     axes[0, 1].set_xlabel('Time')
    #     axes[0, 1].set_ylabel('Price Change')
    #     axes[0, 1].grid(True, alpha=0.3)
        
    #     # ACF
    #     plot_acf(diff_series, lags=lags, ax=axes[1, 0])
    #     axes[1, 0].set_title('Autocorrelation Function (ACF)')
        
    #     # PACF
    #     plot_pacf(diff_series, lags=lags, ax=axes[1, 1])
    #     axes[1, 1].set_title('Partial Autocorrelation Function (PACF)')
        
    #     plt.tight_layout()
    #     plt.savefig('results/arima_diagnostics.png', dpi=150)
    #     plt.close()
        
    #     print(f"\n✅ Diagnostic plots saved to results/arima_diagnostics.png")
    #     print(f"\nHow to interpret:")
    #     print(f"  - ACF: Look for cut-off point to determine q (MA order)")
    #     print(f"  - PACF: Look for cut-off point to determine p (AR order)")
    #     print(f"  - If gradual decay in both: try ARMA model")
    #     print(f"  - Current order: {self.order}")
    
    def fit(self, train_data, column='Close'):
        """
        Fit ARIMA model to training data
        
        Args:
            train_data: DataFrame with training data
            column: Column to predict
        """
        print(f"\n{'='*60}")
        print(f"FITTING ARIMA MODEL")
        print(f"{'='*60}")
        print(f"Order (p, d, q): {self.order}")
        if self.seasonal_order:
            print(f"Seasonal order (P, D, Q, s): {self.seasonal_order}")
        print(f"Training samples: {len(train_data)}")
        
        series = train_data[column]
        
        try:
            if self.seasonal_order:
                # SARIMA model
                self.model = SARIMAX(series, 
                                    order=self.order,
                                    seasonal_order=self.seasonal_order)
            else:
                # Regular ARIMA model
                self.model = ARIMA(series, order=self.order)
            
            self.model_fit = self.model.fit()
            self.is_fitted = True
            
            print(f"\n✅ Model fitted successfully!")
            print(f"\nModel Summary:")
            print(self.model_fit.summary())
            
        except Exception as e:
            print(f"\n❌ Error fitting model: {e}")
            print(f"   Try different order parameters")
            raise
    
    def predict(self, steps=1):
        """
        Predict future values
        
        Args:
            steps: Number of steps ahead to predict
        
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.model_fit.forecast(steps=steps)
        return forecast.values
    
    def predict_rolling(self, test_data, column='Close'):
        """
        Make rolling one-step-ahead predictions on test data
        
        Args:
            test_data: DataFrame with test data
            column: Column to predict
        
        Returns:
            predictions: Array of predictions
            actuals: Array of actual values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        print(f"\n{'='*60}")
        print(f"MAKING ROLLING PREDICTIONS")
        print(f"{'='*60}")
        print(f"Test samples: {len(test_data)}")
        
        predictions = []
        actuals = test_data[column].values
        
        # For each test point, refit model and predict next point
        for i in range(len(test_data)):
            # Make one-step prediction
            pred = self.predict(steps=1)[0]
            predictions.append(pred)
            
            # Update model with actual observed value for next prediction
            if i < len(test_data) - 1:
                self.model_fit = self.model_fit.append([actuals[i]])
            
            if (i + 1) % 100 == 0:
                print(f"  Predicted {i + 1}/{len(test_data)} samples")
        
        print(f"✅ Completed rolling predictions")
        
        return np.array(predictions), actuals
    
    def evaluate(self, predictions, actuals):
        """
        Evaluate model performance
        
        Args:
            predictions: Predicted values
            actuals: Actual values
        
        Returns:
            Dictionary with metrics
        """
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals)**2))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Direction accuracy (did we predict up/down correctly?)
        pred_direction = np.diff(predictions) > 0
        actual_direction = np.diff(actuals) > 0
        direction_accuracy = np.mean(pred_direction == actual_direction) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }
        
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION")
        print(f"{'='*60}")
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")
        
        return metrics


def auto_arima_order(data, column='Close', max_p=5, max_d=2, max_q=5):
    """
    Automatically find best ARIMA order using AIC/BIC
    
    Args:
        data: DataFrame with time series
        column: Column to model
        max_p: Maximum AR order to test
        max_d: Maximum differencing order to test
        max_q: Maximum MA order to test
    
    Returns:
        Best order (p, d, q)
    """
    print(f"\n{'='*60}")
    print(f"AUTO ARIMA ORDER SELECTION")
    print(f"{'='*60}")
    print(f"Testing orders up to p={max_p}, d={max_d}, q={max_q}")
    
    series = data[column]
    best_aic = np.inf
    best_order = None
    
    results = []
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    model_fit = model.fit()
                    aic = model_fit.aic
                    bic = model_fit.bic
                    
                    results.append({
                        'order': (p, d, q),
                        'aic': aic,
                        'bic': bic
                    })
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                    
                    if (p + d + q) % 5 == 0:
                        print(f"  Tested {len(results)} models...")
                        
                except:
                    continue
    
    # Sort by AIC
    results_df = pd.DataFrame(results).sort_values('aic')
    
    print(f"\n✅ Tested {len(results)} models")
    print(f"\nTop 5 models by AIC:")
    print(results_df.head(10).to_string(index=False))
    
    print(f"\n🏆 Best order: {best_order} (AIC: {best_aic:.2f})")
    
    return best_order


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
    plt.close()
    
    print(f"\n✅ Prediction plot saved to results/arima_predictions.png")

# Plot residuals
def plot_residuals(test_data, predictions, column='Close'):
    """Plot residual diagnostics"""

    percentage_within_1_percent = (
        np.mean(
            np.abs((predictions/test_data[column].values - 1)*100)
            <= 1
        )
        * 100
    )
    
    #fig = model_fit.plot_diagnostics(figsize=(15, 10))
    #Plot differences between actual and predicted values
    plt.plot((predictions/test_data[column].values - 1)*100)
    plt.title(
        f"Prediction Error Percentage (Predicted - Actual)/(Actual)*100\n{percentage_within_1_percent:.2f}% of predictions within ±1% error"
    )
    plt.xlabel('Time')
    plt.ylabel('Error (%)')
    plt.grid(True, alpha=0.3)
    # Lines at y = 1 and y = -1
    plt.axhline(y=1, linestyle="--", color="red", label="1% Error Line")
    plt.axhline(y=-1, linestyle="--", color="red", label="-1% Error Line")
    plt.legend()
    plt.savefig('results/arima_prediction_errors.png', dpi=150)
    plt.close()
    
    print(f"✅ Prediction Error Percentage saved to results/arima_prediction_errors.png")


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
