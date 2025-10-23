# # source/__init__.py

# """Stock prediction package"""

# __version__ = '1.0.0'

# # Import modules using relative imports
# from . import stockdataloader
# from . import stockpreprocess
# from . import config

# # Expose commonly used items
# from .stockdataloader import load_stock_data
# from .stockpreprocess import preprocess_stock_data, StockDataset
# from .config import train_test_split_percent, hidden_dim, num_layers, epochs, lr

# __all__ = [
#     'stockdataloader',
#     'stockpreprocess',
#     'config',
#     'load_stock_data',
#     'preprocess_stock_data',
#     'StockDataset']