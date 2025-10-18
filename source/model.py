import config
import torch
import torch.nn as nn
import torch.optim as optim
import stockpreprocess

class QAModel(nn.Module):
    def __init__(self, embedding_dim=config.embedding_dim, hidden_dim=config.hidden_dim, num_layers=config.num_layers):
        super(QAModel, self).__init__()
        
        #self.embedding = nn.Embedding(contexts, embedding_dim, padding_idx=0)
        
        # Unidirectional LSTM for context and target
        self.context_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            #dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.target_lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            #dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.start_output = nn.Linear(hidden_dim * 2, 1)  # concatenate context with target representation
        self.end_output = nn.Linear(hidden_dim * 2, 1)
        
        #self.dropout = nn.Dropout(dropout)
    
    def forward(self, context, target):
        # Embed context and target
        #context_embedded = self.dropout(self.embedding(context))
        #target_embedded = self.dropout(self.embedding(target))
        
        # Process context and target through LSTM
        context_outputs, _ = self.context_lstm(self.embedding(context))
        _, (target_hidden, _) = self.target_lstm(self.embedding(target))
        
        # Get final target representation (last hidden state)
        target_repr = target_hidden[-1]  # shape: [batch_size, hidden_dim]
        
        # Expand target representation to match context length
        target_repr = target_repr.unsqueeze(1).expand(-1, context_outputs.size(1), -1)
        
        # Combine context and target representations
        combined = torch.cat([context_outputs, target_repr], dim=2)
        
        # Get start and end logits
        start_logits = self.start_output(combined).squeeze(-1)
        end_logits = self.end_output(combined).squeeze(-1)
        
        return start_logits, end_logits

# Create model instance
model = QAModel()

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(model)