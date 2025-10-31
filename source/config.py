# train_test_split_percent = 0.8
# hidden_dim=64
# num_layers=1
# epochs=50
# lr = 0.001
# input_size=6  # Number of features in the stock data (e.g., Open, High, Low, Close, Adj Close, Volume)

# config.py
# This file loads config.yaml and makes variables accessible

import yaml
import os


class Config:
    """Configuration manager that loads from YAML and provides attribute access"""

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key, default=None):
        """Get a configuration value using dot notation (e.g., 'model.hidden_dim')"""
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getattr__(self, name):
        """Allow attribute-style access to top-level config sections"""
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        if name in self._config:
            section = self._config[name]
            if isinstance(section, dict):
                return type("ConfigSection", (), section)()
            return section

        raise AttributeError(f"Config has no attribute '{name}'")

    def __repr__(self):
        return f"Config({self.config_path})"

    def display(self):
        """Pretty print the configuration"""
        import json

        print(json.dumps(self._config, indent=2))


# Create a global config instance
config = Config(config_path="config.yaml")

# Variables
train_test_split_percent = config.get("data.train_test_split_percent")
path = config.get("data.data_path")
sequence_length = config.get("data.sequence_length")
batch_size = config.get("data.batch_size")
hidden_dim = config.get("model.hidden_dim")
num_layers = config.get("model.num_layers")
epochs = config.get("training.epochs")
lr = config.get("training.learning_rate")
input_size = config.get("model.input_size")
batch_size = config.get("data.batch_size")
dropout = config.get("model.dropout")
model_type = config.get("model.model_type")
error = config.get("training.error")

# ARIMA parameters
arima_type = config.get("arima.arima_type")
# Regular ARIMA parameters
p = config.get("arima.order.p")
d = config.get("arima.order.d")
q = config.get("arima.order.q")
# SARIMA parameters (set to null for regular ARIMA)
P = config.get("arima.seasonal_order.P")
D = config.get("arima.seasonal_order.D")
Q = config.get("arima.seasonal_order.Q")
s = config.get("arima.seasonal_order.s")
  # Auto ARIMA settings
enabled = config.get("arima.auto_arima.enabled")
max_p = config.get("arima.auto_arima.max_p")
max_d = config.get("arima.auto_arima.max_d")
max_q = config.get("arima.auto_arima.max_q")


if __name__ == "__main__":
    print("Configuration loaded from:", config.config_path)
    print("\nFull configuration:")
    config.display()

    print("\n--- Accessing config values ---")
    print(f"Training epochs: {config.training.epochs}")
    print(f"Model hidden dim: {config.model.hidden_dim}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Train/test split: {config.data.train_test_split_percent}")

    print("\n--- Using module-level variables ---")
    print(f"epochs = {epochs}")
    print(f"hidden_dim = {hidden_dim}")
    print(f"lr = {lr}")
