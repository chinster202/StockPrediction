import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMAResults
import pandas as pd
import config
import arima_model
from celery import Celery

def load_model(ticker_path):
    """
    Load the best ARIMA model from disk.
    """
    try:
        loaded = ARIMAResults.load(f'models/arima_model_{ticker_path}.pkl')
        print("ARIMA model loaded successfully!")
        return loaded
    except Exception as e:
        print(f"Error loading ARIMA model: {e}")
        return None

def import_data():
    stock_files = [f for f in config.tickers]
    print(f"\nFound {len(stock_files)} stock files: {stock_files}")

    for path in stock_files:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {path}")
        print(f"{'='*80}")

        # Get the current date and time as a pandas Timestamp object
        today = pd.to_datetime("today").date()

        # Subtract two years using pd.DateOffset
        # two_years_ago = today - pd.DateOffset(years=2)

        print(f"Today's date: {today}")

        tomorrow = today + pd.DateOffset(days=1)
        print(f"Tomorrow's date: {tomorrow}")

        backdate = pd.to_datetime(today - pd.DateOffset(days=11))

        yf_data = yf.download(path, start=backdate.strftime("%Y-%m-%d"), end=tomorrow.strftime("%Y-%m-%d"))

        # yf_data = yf.download(path, start=two_years_ago.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"))

        print(f'datatype is {type(yf_data)} \n')

        # Save raw CSV with yfinance (includes metadata rows)
        yf_data.to_csv(f"data/{path}_latest.csv")

        # Reload and remove the 2 metadata lines
        cleaned = pd.read_csv(f"data/{path}_latest.csv", skiprows=2)

        # Set Date as index
        cleaned = cleaned.set_index("Date")

        cleaned.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

        # Save the cleaned version
        cleaned.to_csv(f"data/{path}_latest.csv")

        print(f"✅ Downloaded data for {path} from yfinance and saved to data/{path}_latest.csv")

app = Celery('hello', broker='redis://localhost:6379/0')      

@app.task
def main():

    import_data()

    predictions_dict = {}

    for path in config.tickers:

        # Load the arima_model into the StockArima class
        arima = arima_model.StockARIMA()
        # arima.model_fit = load_model(path)
        arima.model_fit = load_model(f'{path}')
        arima.is_fitted = True

        # Read data
        stockdf = pd.read_csv(f"data/{path}_latest.csv")

        # Get last row of data for prediction
        last_row = stockdf.iloc[-1]
        # Get date of last row
        last_date = last_row['Date']

        # Make predictions
        predictions, actuals = arima.predict_rolling(stockdf, column='Close')

        predictions_dict[path] = (last_date, predictions[-1])

        print(f"\nPredicted next 'Close' price for {path} on {last_date}: ${predictions[-1]:.2f}")

    # Read all_stocks_summary.csv
    summary_df = pd.read_csv("results/all_stocks_summary.csv")
    # Put values in a dictionary
    summary_dict = dict(zip(summary_df['stock'], summary_df['optimal_n_days_threshold']))
    print("\nStock Summaries:")
    for ticker, summary in summary_dict.items():
        print(f"  - {ticker}: {summary}")
        
    print("\nAll predictions:")
    for ticker, (date, pred) in predictions_dict.items():
        print(f"  - {ticker}: ${pred:.2f} on {date}")
    print("Predictions successfully made!")

    return predictions_dict, summary_dict

if __name__ == "__main__":

    predictions_dict, summary_dict = main()