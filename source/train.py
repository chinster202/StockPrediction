import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import stockpreprocess
import stockdataloader
import config
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import StockLSTM


# Get preprocessed data

contexts_df, targets_df = stockdataloader.load_stock_data(stockdataloader.path)

train_context_data, train_target_data, val_context_data, val_target_data, train_loader, val_loader = stockpreprocess.preprocess_stock_data(contexts_df, targets_df)

model = StockLSTM()

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        context = batch['context'].float()
        target = batch['target'].float().unsqueeze(1)

        #print(context)
        #print(target)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(context)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            context = batch['context'].float()
            target = batch['target'].float().unsqueeze(1)
            
            output = model(context)
            loss = criterion(output, target)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader), target, output

def train_model(model, train_loader, val_loader, epochs=config.epochs, lr=config.lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_target, val_output = validate(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_stock_model.pth')
            print(f"Best model saved with validation loss: {val_loss:.6f}")
    
    return train_losses, val_losses, val_target, val_output

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

if __name__ == "__main__":
    # Get preprocessed data
    contexts_df, targets_df = stockdataloader.load_stock_data(stockdataloader.path)
    train_context_data, train_target_data, val_context_data, val_target_data, train_loader, val_loader = stockpreprocess.preprocess_stock_data(contexts_df, targets_df)
    
    # Get input size from first batch
    #sample_batch = next(iter(train_loader))
    #seq_len = sample_batch['context'].shape[1]
    #input_size = sample_batch['context'].shape[2]
    #print(f"\nSequence length: {seq_len}, Input size (features): {input_size}")
    
    # Initialize model with config parameters
    model = StockLSTM(
        input_size=config.input_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        #dropout=0.2
    )
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print(f"\nStarting training for {config.epochs} epochs with learning rate {config.lr}")
    train_losses, val_losses, val_target, val_output = train_model(
        model,
        train_loader,
        val_loader,
        epochs=config.epochs,
        lr=config.lr
    )

    print("")
    
    # Plot results
    plot_losses(train_losses, val_losses)
    
    print("\nTraining complete! Best model saved as 'best_stock_model.pth'")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Val Loss: {val_losses[-1]:.6f}")
    print(f"Best Val Loss: {min(val_losses):.6f}")