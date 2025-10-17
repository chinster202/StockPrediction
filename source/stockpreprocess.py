import stockdataloader

train_df, val_df = stockdataloader.load_stock_data(stockdataloader.path)

def preprocess_stock_data(train_df, val_df):

    train_df_no_date = train_df.drop(columns=['Date'])

    contexts = []
    targets = []
    for i in range(len(val_df) - 1):
        contexts.append(train_df_no_date.iloc[i:i+6].values.tolist())
        targets.append(val_df.iloc[i]['Close'])
    
    return contexts, targets

contexts, targets = preprocess_stock_data(train_df, val_df)

print("Preprocessing module loaded.")

if __name__ == "__main__":
    contexts, targets = preprocess_stock_data(train_df, val_df)
    print(f"Number of training contexts: {len(contexts)}")
    print(f"Number of training targets: {len(targets)}")
    print("First context:", contexts[0])
    print("First target:", targets[0])