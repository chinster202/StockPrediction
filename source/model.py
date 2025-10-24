from . import config
import torch.nn as nn


class StockLSTM(nn.Module):
    def __init__(
        self,
        input_size=config.input_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    ):
        super(StockLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=config.dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]

        # Pass through fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)

        return out


# Create model instance
model = StockLSTM()

print(model)
