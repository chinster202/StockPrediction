import yfinance as yf

# test if yfinance can download stock data
ticker = "AAPL"
data = yf.download(ticker, start="2026-01-01", end="2026-02-06")
assert not data.empty, "Downloaded data should not be empty"
assert "Close" in data.columns, "Data should contain 'Close' column"
#assert len(data) == 7, "There should be 7 trading days of data"  # Jan 1-10, 2023 has 7 trading days
print("yfinance download test passed.")
print(data)