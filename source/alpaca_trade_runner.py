#!/usr/bin/env python3
"""
alpaca_trade_runner.py - SIMPLIFIED VERSION
Reads only today's partition which contains multiple days of prediction data
"""

import os
import json
import io
from datetime import datetime, date, timedelta, timezone
import boto3
import pandas as pd
import numpy as np
from decimal import Decimal
from alpaca_trade_api.rest import REST, TimeFrame
import alpaca_trade_api as tradeapi
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Alpaca client
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception:
    TradingClient = None
    MarketOrderRequest = None
    LimitOrderRequest = None
    OrderSide = None
    TimeInForce = None

# -----------------------
# Config
# -----------------------
S3_BUCKET = os.environ.get("S3_BUCKET", "my-stock-model-data")
S3_PREFIX = os.environ.get("S3_PREFIX", "predictions/")
AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")

ALPACA_API_KEY = "PKZGQO5GPJZG6VMRON67VSXDU4"
ALPACA_SECRET = "MLz623CbwLbcq6yBovW5NdUMDLzxDVJHgTe7p7UdRWL"
ALPACA_PAPER = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

api = REST(ALPACA_API_KEY, ALPACA_SECRET, ALPACA_BASE_URL)

N_DAYS_THRESHOLD = 3
INITIAL_CAPITAL_PER_TICKER = 100000.0
STARTING_DAY = "2026-01-16"

PORTFOLIO_STATE_KEY = os.path.join(S3_PREFIX, "portfolio_state.json")
PREDICTIONS_PREFIX = S3_PREFIX
PORTFOLIO_BACKUP_KEY = os.path.join(S3_PREFIX, "portfolio_state_backup.json")

# boto3 client
s3 = boto3.client("s3", region_name=AWS_REGION)

def get_alpaca_client():
    if ALPACA_API_KEY is None or ALPACA_SECRET is None:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET must be set")
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET, paper=ALPACA_PAPER)
    return client

# -----------------------
# SIMPLIFIED: Read only today's partition (contains multi-day data)
# -----------------------
def read_all_predictions_from_s3(bucket=S3_BUCKET, prefix=PREDICTIONS_PREFIX):
    """
    Reads predictions from today's partition only
    Today's partition contains multiple days worth of prediction data
    Handles Hive-style partitioning: run_date=YYYY-MM-DD/
    """
    today_utc = datetime.now(timezone.utc).date()
    today_str = today_utc.strftime("%Y-%m-%d")
    
    # Construct the partition path for today
    partition_prefix = os.path.join(prefix, f"run_date={today_str}/")
    
    print(f"Reading predictions from today's partition: run_date={today_str}")
    print(f"Looking in: s3://{bucket}/{partition_prefix}\n")
    
    paginator = s3.get_paginator("list_objects_v2")
    
    try:
        pages = paginator.paginate(Bucket=bucket, Prefix=partition_prefix)
    except Exception as e:
        print(f"Error listing objects: {e}")
        return pd.DataFrame(columns=["ticker", "date", "prediction"])

    frames = []
    file_count = 0
    
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            
            # Skip directories
            if key.endswith("/"):
                continue
            
            # Skip portfolio state files
            if "portfolio_state" in key:
                continue
            
            # Only process parquet files
            if key.endswith((".parquet", ".snappy.parquet", ".gzip.parquet")):
                print(f"Reading: s3://{bucket}/{key}")
                
                try:
                    obj_body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                    df = pd.read_parquet(io.BytesIO(obj_body))
                    
                    if not df.empty:
                        frames.append(df)
                        file_count += 1
                        print(f"  ✓ Loaded {len(df)} rows")
                    else:
                        print(f"  ⚠ File is empty")
                        
                except Exception as e:
                    print(f"  ✗ Error reading {key}: {e}")
                    continue

    if not frames:
        print(f"\n⚠ No prediction files found in partition run_date={today_str}")
        return pd.DataFrame(columns=["ticker", "date", "prediction"])

    print(f"\n✓ Successfully read {file_count} parquet files")
    
    df_all = pd.concat(frames, ignore_index=True, sort=False)
    
    # Normalize column names
    df_all.columns = [c.strip().lower() for c in df_all.columns]
    
    print(f"Raw columns in data: {df_all.columns.tolist()}")

    # Identify columns
    col_lower = df_all.columns.tolist()
    
    # Find ticker column
    if "ticker" in col_lower:
        ticker_col = "ticker"
    elif "stock" in col_lower:
        ticker_col = "stock"
    elif "symbol" in col_lower:
        ticker_col = "symbol"
    else:
        # Try to find a column with stock ticker patterns
        possible = [c for c in df_all.columns if df_all[c].astype(str).str.match(r"^[A-Z]{1,5}$").any()]
        ticker_col = possible[0] if possible else None

    # Find date column - this is the prediction date within the data
    if "date" in col_lower:
        date_col = "date"
    elif "prediction_date" in col_lower:
        date_col = "prediction_date"
    elif "pred_date" in col_lower:
        date_col = "pred_date"
    else:
        # Look for any date-like column
        date_candidates = [c for c in col_lower if 'date' in c and c != 'run_date']
        date_col = date_candidates[0] if date_candidates else None

    # Find prediction column
    if "prediction" in col_lower:
        pred_col = "prediction"
    elif "pred" in col_lower:
        pred_col = "pred"
    elif "predicted_price" in col_lower:
        pred_col = "predicted_price"
    else:
        # Try to find a numeric column
        numeric = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c])]
        pred_col = numeric[-1] if numeric else None

    # Validate required columns
    if ticker_col is None:
        raise ValueError(f"Could not identify ticker column. Available columns: {df_all.columns.tolist()}")
    if pred_col is None:
        raise ValueError(f"Could not identify prediction column. Available columns: {df_all.columns.tolist()}")
    if date_col is None:
        print(f"⚠ Warning: No date column found in data. Using run_date={today_str} for all rows")
        df_all["date"] = today_utc
        date_col = "date"

    # Build normalized DataFrame
    normalized = pd.DataFrame()
    normalized["ticker"] = df_all[ticker_col].astype(str).str.strip().str.upper()
    normalized["date"] = pd.to_datetime(df_all[date_col]).dt.date
    normalized["prediction"] = pd.to_numeric(df_all[pred_col], errors="coerce")

    # Remove any rows with missing data
    normalized = normalized.dropna(subset=["ticker", "prediction"])
    
    # Sort by ticker and date
    normalized = normalized.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    print(f"\n✓ Total predictions loaded: {len(normalized)}")
    print(f"  Unique tickers: {normalized['ticker'].nunique()}")
    print(f"  Date range: {normalized['date'].min()} to {normalized['date'].max()}")
    
    # Show predictions per ticker with date counts
    print(f"\nPredictions per ticker:")
    for ticker in sorted(normalized['ticker'].unique()):
        ticker_data = normalized[normalized['ticker'] == ticker]
        dates = ticker_data['date'].unique()
        print(f"  {ticker}: {len(dates)} days of data ({dates.min()} to {dates.max()})")
    
    return normalized

# -----------------------
# Portfolio persistence
# -----------------------
def load_portfolio_state(bucket=S3_BUCKET, key=PORTFOLIO_STATE_KEY):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        payload = obj["Body"].read().decode("utf-8")
        state = json.loads(payload)
        print(f"\nLoaded portfolio state from s3://{bucket}/{key}")
        return state
    except s3.exceptions.NoSuchKey:
        print("\nNo existing portfolio state; creating new.")
        return {}
    except Exception as e:
        print(f"\nUnable to read portfolio state from S3: {e}")
        return {}

def save_portfolio_state(state, bucket=S3_BUCKET, key=PORTFOLIO_STATE_KEY):
    try:
        s3.copy_object(Bucket=bucket, CopySource={'Bucket': bucket, 'Key': key}, Key=PORTFOLIO_BACKUP_KEY)
    except Exception:
        pass
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(state, default=str).encode("utf-8"))
    print(f"Saved portfolio state to s3://{bucket}/{key}")

# -----------------------
# Trading signal
# -----------------------
def compute_signal_for_ticker(predictions: pd.Series, n_days=N_DAYS_THRESHOLD):
    """
    predictions: pd.Series sorted by date for a single ticker
    returns: "BUY" / "SELL" / "HOLD"
    """
    if len(predictions) < 2:
        return "HOLD"

    diffs = np.diff(predictions)
    directions = diffs > 0

    if len(directions) < n_days:
        return "HOLD"

    last_dirs = directions[-n_days:]
    if np.all(last_dirs):
        return "BUY"
    if np.all(~last_dirs):
        return "SELL"
    return "HOLD"

# -----------------------
# Get yesterday's closing price
# -----------------------
# def get_yesterday_close_price(ticker):
#     """
#     Fetch yesterday's closing price
#     Uses Alpaca only
#     """
#     try:
#         data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET)
        
#         # Get last 10 daily bars
#         request = StockBarsRequest(
#             symbol_or_symbols=ticker,
#             timeframe=TimeFrame.Day,
#             limit=10
#         )
        
#         bars = data_client.get_stock_bars(request)

#         print(f"  Bars data: {bars.data}")

#         if not bars or not bars.data:
#             raise RuntimeError(f"No price data returned for {ticker}")

#         # Check if ticker exists in response
#         if ticker not in bars.data or len(bars.data[ticker]) == 0:
#             raise RuntimeError(f"No price data returned for {ticker}")
        
#         today = date.today()
        
#         # Find yesterday's bar
#         for bar in reversed(bars.data[ticker]):
#             # print the bar
#             # print(f"    Bar: {bar}")
#             # print(f"    Bar timestamp: {bar.timestamp} Bar Close: {bar.close}")
#             bar_date = bar.timestamp.date()
#             # print(f"    Bar date: {bar_date}, Today: {today}")
#             # if bar_date < today:
#             close_price = float(bar.close)
#             print(f"  Yesterday's close: ${close_price:.2f} ({bar_date})")
#             return close_price, bar_date
        
#         # raise RuntimeError(f"Could not find yesterday's bar for {ticker}")
        
#     except Exception as e:
#         print(f"  ✗ Error fetching price: {e}")
#         raise

def get_yesterday_close_price_alpaca(ticker):
    """
    Debug version to see what we're getting
    """
    try:
        print(f"  Requesting bars for {ticker}...")
        
        bars_response = api.get_bars(
            ticker,
            TimeFrame.Day,
            limit=10
        )
        
        print(f"  Response type: {type(bars_response)}")
        print(f"  Response: {bars_response}")
        
        bars = bars_response.df
        print(f"  DataFrame shape: {bars.shape}")
        print(f"  DataFrame empty: {bars.empty}")
        
        if not bars.empty:
            print(f"  DataFrame head:\n{bars.head()}")
            latest_close = float(bars['close'].iloc[-1])
            latest_date = bars.index[-1].date()
            print(f"  Latest close: ${latest_close:.2f} ({latest_date})")
            return latest_close, latest_date
        else:
            raise RuntimeError(f"Empty dataframe for {ticker}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

def get_yesterday_close_price(ticker):
    """
    Fetch the most recent closing price using yfinance
    Free and reliable - Alpaca API not returning data
    """
    try:
        import yfinance as yf
        
        # Download last 10 days of data
        df = yf.download(ticker, period="10d", progress=False)
        
        if df.empty:
            raise RuntimeError(f"No price data returned for {ticker}")
        
        # Get the most recent close
        latest_close = float(df['Close'].iloc[-1])
        latest_date = df.index[-1].date()
        
        print(f"  Latest close: ${latest_close:.2f} ({latest_date})")
        return latest_close, latest_date
        
    except Exception as e:
        print(f"  ✗ Error fetching price: {e}")
        import traceback
        traceback.print_exc()
        raise

# -----------------------
# Place market order
# -----------------------
# def place_market_order_at_open(client, symbol, side, qty):
#     """
#     Place a market order to execute at today's market open
#     side: 'buy' or 'sell'
#     qty: number of shares
#     Returns order object or raises.
#     """
#     try:
#         if MarketOrderRequest is not None:
#             req = MarketOrderRequest(
#                 symbol=symbol,
#                 qty=qty,
#                 side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
#                 time_in_force=TimeInForce.DAY,
#             )
#             order = client.submit_order(order_data=req)
#             print(f"  ✓ Market order submitted: {order.id}")
#             return order
#         else:
#             # fallback to older alpaca.tradeapi
#             order = client._api.submit_order(
#                 symbol=symbol, 
#                 qty=qty, 
#                 side=side, 
#                 type='market', 
#                 time_in_force='day'
#             )
#             print(f"  ✓ Market order submitted: {order.id}")
#             return order
#     except Exception as e:
#         print(f"  ✗ Order placement failed for {symbol}: {e}")
#         raise

def place_market_order_at_open(client, symbol, side, qty):
    """
    Place a market order to execute at market open
    """
    try:
        if MarketOrderRequest is not None:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                extended_hours=False,  # Don't execute in extended hours, wait for regular market
            )
            order = client.submit_order(order_data=req)
            print(f"  ✓ Market order submitted: {order.id}")
            return order
        else:
            order = client._api.submit_order(
                symbol=symbol, 
                qty=qty, 
                side=side, 
                type='market', 
                time_in_force='day',
                extended_hours=False
            )
            print(f"  ✓ Market order submitted: {order.id}")
            return order
    except Exception as e:
        print(f"  ✗ Order placement failed for {symbol}: {e}")
        raise

# -----------------------
# Main orchestration
# -----------------------
def run_once():
    """
    Main trading loop
    """
    print("\n" + "="*80)
    print("ALPACA TRADING BOT - RUNNING")
    print("="*80)
    print(f"Strategy: {N_DAYS_THRESHOLD}-day consecutive predictions")
    print(f"Pricing: Yesterday's close for decisions")
    print(f"Execution: Market orders at today's open")
    print("="*80 + "\n")
    
    # 1) Load predictions from today's partition (contains multi-day data)
    predictions_df = read_all_predictions_from_s3()
    if predictions_df.empty:
        print("⚠ No predictions found; exiting.")
        return

    # 2) Aggregate predictions per ticker
    grouped = {
        t: g.sort_values("date")["prediction"].values
        for t, g in predictions_df.groupby("ticker")
    }

    # 3) Load portfolio state
    state = load_portfolio_state()
    if "per_ticker" not in state:
        state["per_ticker"] = {}

    client = None
    if TradingClient is not None:
        client = get_alpaca_client()
    else:
        print("⚠ Alpaca client not available; running dry-run.")

    actions_executed = []

    for ticker, preds in grouped.items():
        ticker = ticker.strip()
        
        print(f"\n{'─'*60}")
        print(f"[{ticker}]")
        
        # Ensure portfolio entry exists
        pstate = state["per_ticker"].get(
            ticker,
            {
                "cash": INITIAL_CAPITAL_PER_TICKER,
                "shares": 0.0,
                "history": []
            }
        )

        # Compute signal
        signal = compute_signal_for_ticker(pd.Series(preds), n_days=N_DAYS_THRESHOLD)
        print(f"  Signal: {signal} (based on {len(preds)} days of predictions)")
        
        # If not enough data for signal, skip
        if len(preds) < N_DAYS_THRESHOLD + 1:
            print(f"  ⚠ Need at least {N_DAYS_THRESHOLD + 1} predictions for signal, have {len(preds)}")
            print(f"  ⏸ HOLD - Waiting for more data")
            continue

        # Get yesterday's closing price
        try:
            yesterday_close, close_date = get_yesterday_close_price(ticker)
        except Exception as e:
            print(f"  ✗ Skipping {ticker} - could not get price")
            continue

        # STARTING-DAY LOGIC
        if STARTING_DAY is not None and str(close_date) == STARTING_DAY:
            print(f"  🚀 STARTING DAY detected → investing all initial capital")
            
            cash = float(pstate["cash"])
            if cash > 0:
                qty = int(np.floor(cash / yesterday_close))
                if qty > 0:
                    print(f"  Placing STARTING-DAY BUY: {qty} shares @ market open")
                    
                    order = None
                    if client is not None:
                        try:
                            order = place_market_order_at_open(client, ticker, "buy", qty)
                        except Exception as e:
                            print(f"  ✗ Order failed: {e}")
                    
                    gross = qty * yesterday_close
                    pstate["cash"] = round(cash - gross, 8)
                    pstate["shares"] = pstate.get("shares", 0.0) + qty
                    
                    action_record = {
                        "ticker": ticker,
                        "action": "STARTING_DAY_BUY",
                        "qty": qty,
                        "estimated_price": yesterday_close,
                        "date": str(date.today()),
                        "assumed_filled": True if order else False,
                    }
                    pstate["history"].append(action_record)
                    actions_executed.append(action_record)
            
            pstate["last_updated"] = datetime.now(timezone.utc).isoformat()
            state["per_ticker"][ticker] = pstate
            continue

        # NORMAL TRADING LOGIC
        if signal == "BUY":
            cash = float(pstate["cash"])
            if cash <= 0:
                print(f"  ⚠ No cash available to buy")
            else:
                qty = int(np.floor((cash * 0.999) / yesterday_close))
                if qty <= 0:
                    print(f"  ⚠ Not enough cash for even 1 share at ${yesterday_close:.2f}")
                else:
                    print(f"  📈 BUY signal: {qty} shares @ market open (est. ${yesterday_close:.2f})")
                    
                    order = None
                    if client is not None:
                        try:
                            order = place_market_order_at_open(client, ticker, "buy", qty)
                        except Exception as e:
                            print(f"  ✗ Order failed: {e}")
                    
                    gross = qty * yesterday_close
                    trans_cost = gross * 0.001
                    pstate["cash"] = round(cash - gross - trans_cost, 8)
                    pstate["shares"] = pstate.get("shares", 0.0) + qty
                    
                    action_record = {
                        "ticker": ticker,
                        "action": "BUY",
                        "qty": qty,
                        "estimated_price": yesterday_close,
                        "date": str(date.today()),
                        "assumed_filled": True if order else False,
                    }
                    pstate["history"].append(action_record)
                    actions_executed.append(action_record)

        elif signal == "SELL":
            shares = float(pstate.get("shares", 0.0))
            if shares <= 0:
                print(f"  ⚠ No shares to sell")
            else:
                qty = int(shares)
                print(f"  📉 SELL signal: {qty} shares @ market open (est. ${yesterday_close:.2f})")
                
                order = None
                if client is not None:
                    try:
                        order = place_market_order_at_open(client, ticker, "sell", qty)
                    except Exception as e:
                        print(f"  ✗ Order failed: {e}")
                
                gross = qty * yesterday_close
                trans_cost = gross * 0.001
                net_proceeds = gross - trans_cost
                
                pstate["cash"] = float(pstate.get("cash", 0.0) + net_proceeds)
                pstate["shares"] = 0.0
                
                action_record = {
                    "ticker": ticker,
                    "action": "SELL",
                    "qty": qty,
                    "estimated_price": yesterday_close,
                    "date": str(date.today()),
                    "assumed_filled": True if order else False,
                }
                pstate["history"].append(action_record)
                actions_executed.append(action_record)
        
        else:
            print(f"  ⏸ HOLD - No action taken")

        pstate["last_updated"] = datetime.now(timezone.utc).isoformat()
        state["per_ticker"][ticker] = pstate

    state["last_run"] = datetime.now(timezone.utc).isoformat()
    save_portfolio_state(state)

    # Summary
    print("\n" + "="*80)
    print("TRADING SESSION SUMMARY")
    print("="*80)
    print(f"Actions executed: {len(actions_executed)}")
    
    for action in actions_executed:
        print(f"  {action['action']}: {action['ticker']} - {action['qty']} shares @ ${action['estimated_price']:.2f}")
    
    summary = {
        "run_at": state["last_run"],
        "actions": actions_executed,
        "portfolio_snapshot": {
            t: {
                "cash": state["per_ticker"][t]["cash"],
                "shares": state["per_ticker"][t]["shares"],
            }
            for t in state["per_ticker"]
        },
    }
    
    print("\n" + json.dumps(summary, indent=2))
    print("="*80 + "\n")
    
    return summary

if __name__ == "__main__":
    run_once()
#     # get_yesterday_close_price("AAPL")

# Add this at the bottom of your file temporarily
# if __name__ == "__main__":
#     # Test API access first
#     print("Testing Alpaca API access...")
#     try:
#         account = api.get_account()
#         print(f"✓ Account connected: {account.status}")
#         print(f"  Buying power: ${float(account.buying_power):,.2f}")
        
#         # Test getting bars
#         print("\nTesting bar data access...")
#         test_bars = api.get_bars("AAPL", TimeFrame.Day, limit=5).df
#         print(f"✓ Got {len(test_bars)} bars for AAPL")
#         print(f"  Latest close: ${test_bars['close'].iloc[-1]:.2f}")
#         print(f"  Latest date: {test_bars.index[-1]}")
        
#     except Exception as e:
#         print(f"✗ API test failed: {e}")
#         import traceback
#         traceback.print_exc()
#         exit(1)
    
#     print("\n" + "="*80)
#     run_once()