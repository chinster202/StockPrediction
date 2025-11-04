# import stockpreprocess
# import stockdataloader
# import source.config as config
from . import stockpreprocess
from . import stockdataloader
from . import config
from .model import StockLSTM, StockGRU
from . import train
from .arima_model import StockARIMA, auto_arima_order, plot_predictions, plot_residuals
from .trading_strategy import TradingSimulator, run_parameter_sweep 
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def aggregate_sweep_results(all_sweep_results, buy_hold_results):
    """
    Aggregate parameter sweep results across multiple stocks
    
    Args:
        all_sweep_results: Dictionary with stock paths as keys and sweep DataFrames as values
        buy_hold_results: Dictionary with buy-and-hold results per stock
    
    Returns:
        DataFrame with aggregated results
    """
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS ACROSS ALL STOCKS")
    print(f"{'='*80}")

    # all_sweep_results and buy_hold_results are passed from main()
    
    # Collect all results by n_days_threshold
    n_days_aggregated = {}
    
    for stock_path, sweep_df in all_sweep_results.items():
        print(f"\nProcessing: {stock_path}")
        
        for _, row in sweep_df.iterrows():
            n_days = row['n_days_threshold']
            
            if n_days not in n_days_aggregated:
                n_days_aggregated[n_days] = {
                    'final_values': [],
                    'returns': [],
                    'outperformances': [],
                    'num_trades': [],
                    'transaction_costs': []
                }
            
            n_days_aggregated[n_days]['final_values'].append(row['final_value'])
            n_days_aggregated[n_days]['returns'].append(row['return_percent'])
            n_days_aggregated[n_days]['outperformances'].append(row['outperformance'])
            n_days_aggregated[n_days]['num_trades'].append(row['num_trades'])
            n_days_aggregated[n_days]['transaction_costs'].append(row['transaction_costs'])
    
    # Calculate buy-and-hold totals
    total_buy_hold_value = sum(buy_hold_results.values())
    num_stocks = len(buy_hold_results)
    total_initial = num_stocks * config.initial_capital
    buy_hold_total_return = ((total_buy_hold_value - total_initial) / total_initial) * 100
    
    # Calculate aggregated statistics
    aggregated_results = []
    
    for n_days in sorted(n_days_aggregated.keys()):
        data = n_days_aggregated[n_days]
        
        # Total final value across all stocks (sum of portfolios)
        total_final_value = sum(data['final_values'])
        
        # Total return for the combined portfolio
        total_return = ((total_final_value - total_initial) / total_initial) * 100
        
        # Outperformance vs buy-and-hold
        total_outperformance = total_return - buy_hold_total_return
        
        # Average return across all stocks
        avg_return = np.mean(data['returns'])
        median_return = np.median(data['returns'])
        std_return = np.std(data['returns'])
        
        # Average outperformance
        avg_outperformance = np.mean(data['outperformances'])
        
        # Total and average trades
        total_trades = sum(data['num_trades'])
        avg_trades = np.mean(data['num_trades'])
        
        # Total transaction costs
        total_transaction_costs = sum(data['transaction_costs'])
        
        # Count winning stocks (positive return)
        num_winning = sum(1 for r in data['returns'] if r > 0)
        win_rate = (num_winning / num_stocks) * 100
        
        # Count stocks that beat buy-and-hold
        # buy_hold_returns = []
        # for stock in buy_hold_results.keys():
        #     sweep_df = all_sweep_results[stock]
        #     row = sweep_df[sweep_df['n_days_threshold'] == n_days].iloc[0]
        #     buy_hold_returns.append(row['return_percent'])
        
        num_beat_buy_hold = sum(1 for o in data['outperformances'] if o > 0)
        beat_buy_hold_rate = (num_beat_buy_hold / num_stocks) * 100
        
        aggregated_results.append({
            'n_days_threshold': n_days,
            'num_stocks': num_stocks,
            'total_final_value': total_final_value,
            'total_return': total_return,
            'total_outperformance_vs_buy_hold': total_outperformance,
            'avg_return': avg_return,
            'median_return': median_return,
            'std_return': std_return,
            'avg_outperformance': avg_outperformance,
            'total_trades': total_trades,
            'avg_trades_per_stock': avg_trades,
            'total_transaction_costs': total_transaction_costs,
            'num_winning_stocks': num_winning,
            'win_rate': win_rate,
            'num_beat_buy_hold': num_beat_buy_hold,
            'beat_buy_hold_rate': beat_buy_hold_rate
        })
    
    results_df = pd.DataFrame(aggregated_results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("AGGREGATED PARAMETER SWEEP RESULTS")
    print(f"{'='*80}")
    print(f"\nNumber of stocks: {num_stocks}")
    print(f"Initial capital per stock: $100,000")
    print(f"Total initial capital: ${total_initial:,.2f}")
    print(f"\n{'='*80}")
    print("BUY & HOLD BENCHMARK")
    print(f"{'='*80}")
    print(f"Total Buy & Hold Value: ${total_buy_hold_value:,.2f}")
    print(f"Total Buy & Hold Return: {buy_hold_total_return:.2f}%")
    print(f"Buy & Hold Profit: ${total_buy_hold_value - total_initial:,.2f}")
    
    print(f"\n{results_df.to_string(index=False)}")
    
    # Find best parameter
    best_idx = results_df['total_final_value'].idxmax()
    best_n_days = results_df.loc[best_idx, 'n_days_threshold']
    #best_n_days = results_df.iloc[best_idx]['n_days_threshold'] 
    best_total_value = results_df.loc[best_idx, 'total_final_value']
    best_total_return = results_df.loc[best_idx, 'total_return']
    best_outperformance = results_df.loc[best_idx, 'total_outperformance_vs_buy_hold']
    
    print(f"\n{'='*80}")
    print(f"OPTIMAL PARAMETER")
    print(f"{'='*80}")
    print(f"🏆 Best n_days_threshold: {best_n_days}")
    print(f"   Total Final Value: ${best_total_value:,.2f}")
    print(f"   Total Return: {best_total_return:.2f}%")
    print(f"   Outperformance vs Buy & Hold: {best_outperformance:+.2f}%")
    print(f"   Win Rate: {results_df.loc[best_idx, 'win_rate']:.1f}%")
    print(f"   Beat Buy & Hold Rate: {results_df.loc[best_idx, 'beat_buy_hold_rate']:.1f}%")
    
    # Add buy-and-hold comparison
    print(f"\n{'='*80}")
    print("COMPARISON: BEST STRATEGY vs BUY & HOLD")
    print(f"{'='*80}")
    print(f"{'Strategy':<30} {'Value':<20} {'Return':<15} {'Profit':<20}")
    print(f"{'-'*80}")
    print(f"{'Buy & Hold Benchmark':<30} ${total_buy_hold_value:>18,.2f} {buy_hold_total_return:>13.2f}% ${total_buy_hold_value - total_initial:>18,.2f}")
    print(f"{'Best Trading Strategy':<30} ${best_total_value:>18,.2f} {best_total_return:>13.2f}% ${best_total_value - total_initial:>18,.2f}")
    print(f"{'-'*80}")
    print(f"{'Difference':<30} ${best_total_value - total_buy_hold_value:>18,.2f} {best_outperformance:>13.2f}%")
    
    return results_df, buy_hold_total_return


def plot_aggregated_results(results_df, all_sweep_results, buy_hold_total_return):
    """
    Plot aggregated results across all stocks with buy-and-hold comparison
    
    Args:
        results_df: Aggregated results DataFrame
        all_sweep_results: Dictionary of individual stock results
        buy_hold_total_return: Total buy-and-hold return percentage
    """
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Total final value across all stocks vs buy-and-hold
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results_df['n_days_threshold'], results_df['total_final_value'], 
             marker='o', linewidth=2, markersize=8, label='Trading Strategy')
    
    # Add buy-and-hold benchmark line
    num_stocks = results_df['num_stocks'].iloc[0]
    total_initial = num_stocks * 100000
    buy_hold_value = total_initial * (1 + buy_hold_total_return / 100)
    ax1.axhline(y=buy_hold_value, color='orange', linestyle='--', linewidth=2, 
                label=f'Buy & Hold: ${buy_hold_value/1000:.0f}K')
    ax1.axhline(y=total_initial, color='r', linestyle=':', linewidth=1.5,
                label=f'Initial Capital: ${total_initial/1000:.0f}K')
    
    ax1.set_xlabel('n_days_threshold')
    ax1.set_ylabel('Total Final Value ($)')
    ax1.set_title('Total Portfolio Value: Strategy vs Buy & Hold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Plot 2: Total return comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(results_df['n_days_threshold'], results_df['total_return'], 
             marker='o', linewidth=2, markersize=8, label='Trading Strategy', color='blue')
    ax2.axhline(y=buy_hold_total_return, color='orange', linestyle='--', linewidth=2,
                label=f'Buy & Hold: {buy_hold_total_return:.2f}%')
    ax2.axhline(y=0, color='r', linestyle=':', alpha=0.5)
    ax2.set_xlabel('n_days_threshold')
    ax2.set_ylabel('Return (%)')
    ax2.set_title('Total Return Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Outperformance vs buy-and-hold
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['green' if x > 0 else 'red' for x in results_df['total_outperformance_vs_buy_hold']]
    ax3.bar(results_df['n_days_threshold'], results_df['total_outperformance_vs_buy_hold'], 
            color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('n_days_threshold')
    ax3.set_ylabel('Outperformance (%)')
    ax3.set_title('Outperformance vs Buy & Hold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Average return with std dev
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(results_df['n_days_threshold'], results_df['avg_return'], 
             marker='o', linewidth=2, markersize=8, label='Avg Return')
    ax4.fill_between(results_df['n_days_threshold'],
                     results_df['avg_return'] - results_df['std_return'],
                     results_df['avg_return'] + results_df['std_return'],
                     alpha=0.2, label='±1 Std Dev')
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('n_days_threshold')
    ax4.set_ylabel('Return (%)')
    ax4.set_title('Average Return per Stock')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Win rates comparison
    ax5 = fig.add_subplot(gs[1, 1])
    x = results_df['n_days_threshold']
    width = 0.35
    ax5.bar(x - width/2, results_df['win_rate'], width, label='Profitable (%)', alpha=0.7, color='green')
    ax5.bar(x + width/2, results_df['beat_buy_hold_rate'], width, label='Beat Buy & Hold (%)', alpha=0.7, color='blue')
    ax5.axhline(y=50, color='gray', linestyle='--', label='50% Baseline')
    ax5.set_xlabel('n_days_threshold')
    ax5.set_ylabel('Percentage (%)')
    ax5.set_title('Success Rates')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim([0, 100])
    
    # Plot 6: Total trading activity
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(results_df['n_days_threshold'], results_df['total_trades'], 
             marker='o', linewidth=2, markersize=8, color='orange')
    ax6.set_xlabel('n_days_threshold')
    ax6.set_ylabel('Total Trades')
    ax6.set_title('Total Trading Activity')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Transaction costs vs profit
    ax7 = fig.add_subplot(gs[2, 0])
    profit = results_df['total_final_value'] - total_initial
    ax7.scatter(results_df['total_transaction_costs'], profit, s=100, alpha=0.7)
    for i, n_days in enumerate(results_df['n_days_threshold']):
        ax7.annotate(f"{int(n_days)}", 
                    (results_df['total_transaction_costs'].iloc[i], profit.iloc[i]),
                    textcoords="offset points", xytext=(5,5), ha='center')
    ax7.set_xlabel('Total Transaction Costs ($)')
    ax7.set_ylabel('Total Profit ($)')
    ax7.set_title('Transaction Costs vs Profit')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 8: Individual stock returns distribution (box plot)
    ax8 = fig.add_subplot(gs[2, 1:])
    
    # Prepare data for box plot
    returns_by_n_days = []
    n_days_labels = []
    
    for n_days in sorted(results_df['n_days_threshold'].unique()):
        returns = []
        for stock_path, sweep_df in all_sweep_results.items():
            row = sweep_df[sweep_df['n_days_threshold'] == n_days]
            if not row.empty:
                returns.append(row['return_percent'].values[0])
        returns_by_n_days.append(returns)
        n_days_labels.append(str(int(n_days)))
    
    bp = ax8.boxplot(returns_by_n_days, labels=n_days_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax8.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax8.axhline(y=buy_hold_total_return, color='orange', linestyle='--', linewidth=2,
                label=f'Avg Buy & Hold: {buy_hold_total_return:.2f}%')
    ax8.set_xlabel('n_days_threshold')
    ax8.set_ylabel('Return (%)')
    ax8.set_title('Distribution of Returns Across All Stocks')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Plot 9: Cumulative comparison table
    ax9 = fig.add_subplot(gs[3, :])
    ax9.axis('tight')
    ax9.axis('off')
    
    # Create summary table
    best_idx = results_df['total_final_value'].idxmax()
    
    table_data = [
        ['Metric', 'Buy & Hold', 'Best Strategy', 'Difference'],
        ['Initial Investment', f'${total_initial:,.0f}', f'${total_initial:,.0f}', '-'],
        ['Final Value', f'${buy_hold_value:,.0f}', f'${results_df.loc[best_idx, "total_final_value"]:,.0f}', 
         f'${results_df.loc[best_idx, "total_final_value"] - buy_hold_value:+,.0f}'],
        ['Return', f'{buy_hold_total_return:.2f}%', f'{results_df.loc[best_idx, "total_return"]:.2f}%',
         f'{results_df.loc[best_idx, "total_outperformance_vs_buy_hold"]:+.2f}%'],
        ['# Trades', '2', f'{results_df.loc[best_idx, "total_trades"]:.0f}', 
         f'{results_df.loc[best_idx, "total_trades"] - 2:+.0f}'],
        ['Transaction Costs', f'${total_initial * 0.001 * 2:,.0f}', 
         f'${results_df.loc[best_idx, "total_transaction_costs"]:,.0f}',
         f'${results_df.loc[best_idx, "total_transaction_costs"] - (total_initial * 0.001 * 2):+,.0f}'],
        ['Best n_days', '-', f'{results_df.loc[best_idx, "n_days_threshold"]:.0f}', '-']
    ]
    
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color difference column
    for i in range(1, len(table_data)):
        if table_data[i][3] != '-':
            if '+' in str(table_data[i][3]):
                table[(i, 3)].set_facecolor('#90EE90')
            elif '-' in str(table_data[i][3]) and table_data[i][3] != '-':
                table[(i, 3)].set_facecolor('#FFB6C6')
    
    plt.savefig('results/aggregated_sweep_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Aggregated results plot saved to results/aggregated_sweep_results.png")

def main():

    all_sweep_results = {}
    all_stocks_summary = []
    buy_hold_results = {}  # Store buy-and-hold values per stock


    stock_files = [f for f in os.listdir("data") if f.endswith('.csv')]
    print(f"\nFound {len(stock_files)} stock files: {stock_files}")

    for path in stock_files:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {path}")
        print(f"{'='*80}")

        # Get preprocessed data
        stockdf, split_idx = stockdataloader.load_stock_data(f"data/{path}")

        if config.model == "RNN":
            (
                _,
                _,
                _,
                _,
                train_loader,
                val_loader,
                means,  # From TRAINING data only
                stds,  # From TRAINING data only
                train_df,
            ) = stockpreprocess.preprocess_stock_data(stockdf, split_idx)

            # Get input size from first batch
            # sample_batch = next(iter(train_loader))
            # seq_len = sample_batch['context'].shape[1]
            # input_size = sample_batch['context'].shape[2]
            # print(f"\nSequence length: {seq_len}, Input size (features): {input_size}")

            # Initialize model with config parameters
            if config.model_type == "StockGRU":
                model = StockGRU(
                    input_size=config.input_size,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                )
            else:
                model = StockLSTM(
                    input_size=config.input_size,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                )

            print("\nModel architecture:")
            print(model)
            print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

            # Plot denormalized training contexts before training
            train.plot_train_contexts(train_df)

            # Train model
            print(
                f"\nStarting training for {config.epochs} epochs with learning rate {config.lr}"
            )
            train_losses, val_losses, val_target, val_output = train.train_model(
                model, train_loader, val_loader, epochs=config.epochs, lr=config.lr
            )

            print("")

            # Plot results
            train.plot_losses(train_losses, val_losses)

            # Plot predictions with denormalization
            train.plot_predictions(val_target, val_output, means, stds)

            print("\nTraining complete! Best model saved as 'best_stock_model.pth'")
            print(f"Final Train Loss: {train_losses[-1]:.6f}")
            print(f"Final Val Loss: {val_losses[-1]:.6f}")
            print(f"Best Val Loss: {min(val_losses):.6f}")

        else:
            print("=" * 60)
            print("ARIMA STOCK PRICE PREDICTION")
            print("=" * 60)

            # Split into train and test
            train_df = stockdf.iloc[:split_idx].copy()
            test_df = stockdf.iloc[split_idx:].copy()

            column = "Close"

            print(f"\nPredicting column: {column}")
            print(f"Training samples: {len(train_df)}")
            print(f"Test samples: {len(test_df)}")

            # Determine seasonal order based on config
            if config.arima_type == "SARIMA":
                seasonal_order = (config.P, config.D, config.Q, config.s)
            else:
                seasonal_order = None

            # Initialize ARIMA model
            arima = StockARIMA(
                order=(config.p, config.d, config.q), seasonal_order=seasonal_order
            )
            # (p, d, q)
            # seasonal_order=(0, 1, 1, 21)  # Only seasonal MA component with seasonality of 21 (trading days in a month)
            # seasonal_order=(2, 1, 2, 21)  # Both seasonal AR and MA components with seasonality of 21

            # You can modify the order and seasonal_order as needed.
            # p = AR order (autoregressive)
            # d = differencing order
            # q = MA order (moving average)
            # P = Seasonal AR order
            # D = Seasonal differencing order
            # Q = Seasonal MA order
            # s = Seasonal period (e.g., 12 for monthly data, 252 for daily stock data)

            # Check stationarity
            arima.check_stationarity(train_df, column=column)

            # Plot diagnostics to help choose p and q
            # arima.plot_diagnostics(train_df, column=column, lags=40)

            # Option 1: Use auto ARIMA to find best order
            print(f"\n{'=' * 60}")
            print("FINDING OPTIMAL ARIMA ORDER")
            print(f"{'=' * 60}")
            user_choice = input("Use auto ARIMA to find best order? (y/n): ").lower()

            if user_choice == "y":
                best_order = auto_arima_order(
                    train_df, column=column, max_p=5, max_d=2, max_q=5
                )
                arima = StockARIMA(order=best_order, seasonal_order=seasonal_order)

            # Fit model
            arima.fit(train_df, column=column)

            # Plot residuals
            # plot_residuals(arima.model_fit)

            # Make rolling predictions on test set
            predictions, actuals = arima.predict_rolling(test_df, column=column)

            # Evaluate
            metrics = arima.evaluate(predictions, actuals)

            # Calculate buy-and-hold value for this stock
            initial_capital = config.initial_capital
            transaction_fee = config.transaction_fee
            buy_hold_shares = (initial_capital * (1 - transaction_fee)) / actuals[0]
            buy_hold_value = buy_hold_shares * actuals[-1]
            buy_hold_results[path] = buy_hold_value

            print(f"\n📊 Buy & Hold Benchmark for {path}:")
            print(f"   Initial: ${initial_capital:,.2f}")
            print(f"   Final: ${buy_hold_value:,.2f}")
            print(f"   Return: {((buy_hold_value - initial_capital) / initial_capital * 100):.2f}%")

            # Show sample predictions
            # compare_predictions_sample(actuals, predictions, n_samples=10)

            # Plot predictions
            plot_predictions(train_df, test_df, predictions, column=column)

            # Plot residuals
            plot_residuals(test_df, predictions, column=column)

            print("\n" + "=" * 60)
            print("ARIMA MODELING COMPLETE!")
            print("=" * 60)
            print(f"\nModel Order: {arima.order}")
            print(f"MAE: ${metrics['mae']:.2f}")
            print(f"RMSE: ${metrics['rmse']:.2f}")
            print(f"MAPE: {metrics['mape']:.2f}%")
            print(f"Direction Accuracy: {metrics['direction_accuracy']:.2f}%")

            # Run trading simulation
            print("\n" + "=" * 80)
            print("TRADING SIMULATION")
            print("=" * 80)

            simulator = TradingSimulator(
                initial_capital=config.initial_capital,
                n_days_threshold=config.n_days_threshold,
                transaction_fee=config.transaction_fee,
            )

            trading_results = simulator.simulate(actuals, predictions)
            simulator.plot_results(actuals)

            # Optional: Run parameter sweep to find best n_days_threshold
            print("\n" + "=" * 80)
            run_sweep = input(
                "Run parameter sweep to find optimal n_days_threshold? (y/n): "
            ).lower()

            if run_sweep == "y":
                sweep_results = run_parameter_sweep(
                    actuals, predictions, n_days_range=range(1, 11), initial_capital=100000
                )
            
            all_sweep_results[path] = sweep_results

            all_stocks_summary.append({
                'stock': path,
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'mape': metrics['mape'],
                'direction_accuracy': metrics['direction_accuracy'],
                'buy_hold_value': buy_hold_value,
                'buy_hold_return': ((buy_hold_value - initial_capital) / initial_capital * 100)
            })

    # After all stocks processed, aggregate sweep results
    # After processing all stocks, aggregate results
    if all_sweep_results:
        print(f"\n{'='*80}")
        print("ALL STOCKS PROCESSED - AGGREGATING RESULTS")
        print(f"{'='*80}")
        
        # Aggregate sweep results with buy-and-hold comparison
        aggregated_df, buy_hold_total_return = aggregate_sweep_results(all_sweep_results, buy_hold_results)
                
        # Plot aggregated results
        plot_aggregated_results(aggregated_df, all_sweep_results, buy_hold_total_return)
        
        # Save aggregated results to CSV
        aggregated_df.to_csv('results/aggregated_sweep_results.csv', index=False)
        print(f"\n✅ Aggregated results saved to results/aggregated_sweep_results.csv")
        
        # Save individual stock summaries
        summary_df = pd.DataFrame(all_stocks_summary)

        # Add optimal strategy results for each stock to summary
        optimal_strategies = []
        for stock_path, sweep_df in all_sweep_results.items():
            best_idx = aggregated_df['total_final_value'].idxmax() #Use this code to get the optimal threshold for each stock -> #sweep_df['final_value'].idxmax()
            optimal_strategies.append({
                'stock': stock_path,
                'optimal_n_days_threshold': sweep_df.loc[best_idx, 'n_days_threshold'],
                'optimal_final_value': sweep_df.loc[best_idx, 'final_value'],
                'optimal_return_percent': sweep_df.loc[best_idx, 'return_percent']
            })
        optimal_df = pd.DataFrame(optimal_strategies)
        summary_df = summary_df.merge(optimal_df, on='stock', how='left')
        summary_df.to_csv('results/all_stocks_summary.csv', index=False)
        print(f"✅ Stock summaries saved to results/all_stocks_summary.csv")
        
        # Print final summary
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"\nProcessed {len(stock_files)} stocks")
        print(f"\nPrediction Performance:")
        print(summary_df.to_string(index=False))
        
        print(f"\n{'='*80}")
        print("BEST TRADING STRATEGY ACROSS ALL STOCKS")
        print(f"{'='*80}")
        best_idx = aggregated_df['total_final_value'].idxmax()
        print(f"\nOptimal n_days_threshold: {aggregated_df.loc[best_idx, 'n_days_threshold']}")
        print(f"Total Initial Investment: ${aggregated_df.loc[best_idx, 'num_stocks'] * 100000:,.2f}")
        print(f"Total Final Value: ${aggregated_df.loc[best_idx, 'total_final_value']:,.2f}")
        print(f"Total Profit: ${aggregated_df.loc[best_idx, 'total_final_value'] - (aggregated_df.loc[best_idx, 'num_stocks'] * 100000):,.2f}")
        print(f"Average Return per Stock: {aggregated_df.loc[best_idx, 'avg_return']:.2f}%")
        print(f"Win Rate: {aggregated_df.loc[best_idx, 'win_rate']:.1f}%")
        
        return aggregated_df, all_sweep_results, buy_hold_results
    else:
        print("\n⚠️  No parameter sweep results collected")
        return None, None


if __name__ == "__main__":
    aggregated_results, individual_results, buy_hold_results = main()
