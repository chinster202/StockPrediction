# source/trading_strategy.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TradingSimulator:
    """Simulate trading strategy based on model predictions"""

    def __init__(
        self, initial_capital=100000, n_days_threshold=3, transaction_fee=0.001
    ):
        """
        Initialize trading simulator

        Args:
            initial_capital: Starting capital in dollars
            n_days_threshold: Number of consecutive days of predictions to trigger action
            transaction_fee: Transaction fee as fraction (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.n_days_threshold = n_days_threshold
        self.transaction_fee = transaction_fee

        # Trading state
        self.cash = initial_capital
        self.shares = 0
        self.position = "out"  # 'in' or 'out'

        # History
        self.history = []

    def reset(self):
        """Reset simulator to initial state"""
        self.cash = self.initial_capital
        self.shares = 0
        self.position = "out"
        self.history = []

    def buy(self, price, date):
        """Buy stock with all available cash"""
        if self.cash > 0:
            # Calculate shares we can buy (accounting for transaction fee)
            effective_cash = self.cash * (1 - self.transaction_fee)
            shares_to_buy = effective_cash / price

            self.shares += shares_to_buy
            self.cash = 0
            self.position = "in"

            return {
                "date": date,
                "action": "BUY",
                "price": price,
                "shares": shares_to_buy,
                "transaction_cost": self.cash * self.transaction_fee,
                "total_value": self.get_portfolio_value(price),
            }
        return None

    def sell(self, price, date):
        """Sell all shares"""
        if self.shares > 0:
            # Calculate cash from sale (accounting for transaction fee)
            gross_proceeds = self.shares * price
            transaction_cost = gross_proceeds * self.transaction_fee
            net_proceeds = gross_proceeds - transaction_cost

            self.cash += net_proceeds
            sold_shares = self.shares
            self.shares = 0
            self.position = "out"

            return {
                "date": date,
                "action": "SELL",
                "price": price,
                "shares": sold_shares,
                "transaction_cost": transaction_cost,
                "total_value": self.get_portfolio_value(price),
            }
        return None

    def hold(self, price, date):
        """Hold current position"""
        return {
            "date": date,
            "action": "HOLD",
            "price": price,
            "shares": self.shares,
            "transaction_cost": 0,
            "total_value": self.get_portfolio_value(price),
        }

    def get_portfolio_value(self, current_price):
        """Get total portfolio value"""
        return self.cash + (self.shares * current_price)

    def simulate(self, actual_prices, predicted_prices, dates=None):
        """
        Simulate trading strategy

        Args:
            actual_prices: Array of actual stock prices
            predicted_prices: Array of predicted stock prices
            dates: Optional array of dates

        Returns:
            Dictionary with simulation results
        """
        self.reset()

        if dates is None:
            dates = list(range(len(actual_prices)))

        print(f"\n{'=' * 80}")
        print("TRADING SIMULATION")
        print(f"{'=' * 80}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Consecutive Days Threshold: {self.n_days_threshold}")
        print(f"Transaction Fee: {self.transaction_fee * 100:.2f}%")
        print(f"Validation Period: {len(actual_prices)} days")
        print(f"{'=' * 80}\n")

        # Calculate predicted direction (up or down)
        predicted_directions = np.diff(predicted_prices) > 0  # True = up, False = down

        # Start fully invested (buy at beginning)
        action = self.buy(actual_prices[0], dates[0])
        if action:
            self.history.append(action)
            print(f"Day 0: INITIAL BUY at ${actual_prices[0]:.2f}")
            print(f"       Shares: {self.shares:.2f}, Cash: ${self.cash:.2f}")

        # Simulate trading day by day
        for i in range(1, len(actual_prices)):
            current_price = actual_prices[i]

            # Check if we have enough history to look back
            if i < self.n_days_threshold:
                action = self.hold(current_price, dates[i])
                self.history.append(action)
                continue

            # Look back at last n_days_threshold predictions
            recent_directions = predicted_directions[i - self.n_days_threshold : i]

            # Check for n consecutive down days
            if np.all(~recent_directions):  # All False (all down)
                if self.position == "in":
                    action = self.sell(current_price, dates[i])
                    if action:
                        print(
                            f"\nDay {i}: SELL SIGNAL - {self.n_days_threshold} consecutive down predictions"
                        )
                        print(
                            f"       Price: ${current_price:.2f}, Shares sold: {action['shares']:.2f}"
                        )
                        print(
                            f"       Cash: ${self.cash:.2f}, Portfolio Value: ${action['total_value']:,.2f}"
                        )
                else:
                    action = self.hold(current_price, dates[i])

            # Check for n consecutive up days
            elif np.all(recent_directions):  # All True (all up)
                if self.position == "out":
                    action = self.buy(current_price, dates[i])
                    if action:
                        print(
                            f"\nDay {i}: BUY SIGNAL - {self.n_days_threshold} consecutive up predictions"
                        )
                        print(
                            f"       Price: ${current_price:.2f}, Shares bought: {action['shares']:.2f}"
                        )
                        print(
                            f"       Cash: ${self.cash:.2f}, Portfolio Value: ${action['total_value']:,.2f}"
                        )
                else:
                    action = self.hold(current_price, dates[i])

            # Otherwise hold
            else:
                action = self.hold(current_price, dates[i])

            if action:
                self.history.append(action)

        # Calculate final results
        final_price = actual_prices[-1]
        final_value = self.get_portfolio_value(final_price)

        # If still holding shares, show what they're worth
        if self.shares > 0:
            print(f"\nFinal position: IN (holding {self.shares:.2f} shares)")
        else:
            print("\nFinal position: OUT (all cash)")

        # Calculate buy-and-hold benchmark
        buy_hold_shares = (
            self.initial_capital * (1 - self.transaction_fee)
        ) / actual_prices[0]
        buy_hold_value = buy_hold_shares * final_price

        # Calculate total transaction costs
        total_transaction_costs = sum(h["transaction_cost"] for h in self.history)

        # Count trades
        num_buys = sum(1 for h in self.history if h["action"] == "BUY")
        num_sells = sum(1 for h in self.history if h["action"] == "SELL")

        # Calculate returns
        strategy_return = (
            (final_value - self.initial_capital) / self.initial_capital
        ) * 100
        buy_hold_return = (
            (buy_hold_value - self.initial_capital) / self.initial_capital
        ) * 100
        outperformance = strategy_return - buy_hold_return

        results = {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "final_cash": self.cash,
            "final_shares": self.shares,
            "final_position": self.position,
            "profit_loss": final_value - self.initial_capital,
            "return_percent": strategy_return,
            "buy_hold_value": buy_hold_value,
            "buy_hold_return": buy_hold_return,
            "outperformance": outperformance,
            "num_trades": num_buys + num_sells,
            "num_buys": num_buys,
            "num_sells": num_sells,
            "total_transaction_costs": total_transaction_costs,
            "history": self.history,
        }

        self.print_summary(results)

        return results

    def print_summary(self, results):
        """Print trading simulation summary"""
        print(f"\n{'=' * 80}")
        print("TRADING SIMULATION RESULTS")
        print(f"{'=' * 80}")
        print(f"Initial Capital:              ${results['initial_capital']:>15,.2f}")
        print(f"Final Portfolio Value:        ${results['final_value']:>15,.2f}")
        print(f"  - Cash:                     ${results['final_cash']:>15,.2f}")
        print(
            f"  - Stock Value:              ${results['final_shares'] * self.history[-1]['price']:>15,.2f}"
        )
        print(f"\nProfit/Loss:                  ${results['profit_loss']:>15,.2f}")
        print(f"Return:                       {results['return_percent']:>15.2f}%")
        print(f"\n{'=' * 80}")
        print("BENCHMARK COMPARISON")
        print(f"{'=' * 80}")
        print(f"Buy & Hold Value:             ${results['buy_hold_value']:>15,.2f}")
        print(f"Buy & Hold Return:            {results['buy_hold_return']:>15.2f}%")
        print(f"Strategy Outperformance:      {results['outperformance']:>15.2f}%")
        print(f"\n{'=' * 80}")
        print("TRADING ACTIVITY")
        print(f"{'=' * 80}")
        print(f"Total Trades:                 {results['num_trades']:>15d}")
        print(f"  - Buys:                     {results['num_buys']:>15d}")
        print(f"  - Sells:                    {results['num_sells']:>15d}")
        print(
            f"Total Transaction Costs:      ${results['total_transaction_costs']:>15,.2f}"
        )
        print(f"{'=' * 80}")

    def plot_results(self, actual_prices, dates=None):
        """Plot portfolio value over time"""

        if dates is None:
            dates = list(range(len(actual_prices)))

        # Extract portfolio values from history
        portfolio_values = [h["total_value"] for h in self.history]
        history_dates = [h["date"] for h in self.history]

        # Calculate buy and hold values
        buy_hold_shares = (
            self.initial_capital * (1 - self.transaction_fee)
        ) / actual_prices[0]
        buy_hold_values = buy_hold_shares * actual_prices

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        # Plot 1: Portfolio value over time
        ax1.plot(history_dates, portfolio_values, label="Trading Strategy", linewidth=2)
        ax1.plot(
            dates, buy_hold_values, label="Buy & Hold", linewidth=2, linestyle="--"
        )
        ax1.axhline(
            y=self.initial_capital, color="gray", linestyle=":", label="Initial Capital"
        )
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title("Portfolio Value: Trading Strategy vs Buy & Hold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"${x / 1000:.0f}K")
        )

        # Plot 2: Stock price with buy/sell markers
        ax2.plot(dates, actual_prices, label="Stock Price", color="black", alpha=0.5)

        # Mark buy and sell points
        for h in self.history:
            if h["action"] == "BUY":
                ax2.scatter(
                    h["date"],
                    h["price"],
                    color="green",
                    marker="^",
                    s=100,
                    zorder=5,
                    label="Buy" if h == self.history[0] else "",
                )
            elif h["action"] == "SELL":
                ax2.scatter(
                    h["date"],
                    h["price"],
                    color="red",
                    marker="v",
                    s=100,
                    zorder=5,
                    label="Sell"
                    if "Sell" not in ax2.get_legend_handles_labels()[1]
                    else "",
                )

        ax2.set_xlabel("Time")
        ax2.set_ylabel("Stock Price ($)")
        ax2.set_title("Stock Price with Buy/Sell Signals")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Position over time (in or out)
        positions = [
            1 if h["action"] in ["BUY", "HOLD"] and h["shares"] > 0 else 0
            for h in self.history
        ]
        ax3.fill_between(history_dates, 0, positions, alpha=0.3, label="In Market")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Position")
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(["OUT", "IN"])
        ax3.set_title("Market Position Over Time")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/trading_simulation.png", dpi=150)

        print("\n✅ Trading simulation plot saved to results/trading_simulation.png")


def run_parameter_sweep(
    actual_prices,
    predicted_prices,
    dates=None,
    n_days_range=range(1, 11),
    initial_capital=100000,
):
    """
    Test different n_days_threshold values to find optimal parameter

    Args:
        actual_prices: Array of actual prices
        predicted_prices: Array of predicted prices
        dates: Optional dates
        n_days_range: Range of n_days_threshold values to test
        initial_capital: Starting capital

    Returns:
        DataFrame with results for each parameter value
    """
    print(f"\n{'=' * 80}")
    print(
        f"PARAMETER SWEEP: Testing n_days_threshold from {min(n_days_range)} to {max(n_days_range)}"
    )
    print(f"{'=' * 80}\n")

    results = []

    for n_days in n_days_range:
        print(f"\nTesting n_days_threshold = {n_days}...")

        simulator = TradingSimulator(
            initial_capital=initial_capital,
            n_days_threshold=n_days,
            transaction_fee=0.001,
        )

        result = simulator.simulate(actual_prices, predicted_prices, dates)

        results.append(
            {
                "n_days_threshold": n_days,
                "final_value": result["final_value"],
                "return_percent": result["return_percent"],
                "outperformance": result["outperformance"],
                "num_trades": result["num_trades"],
                "transaction_costs": result["total_transaction_costs"],
            }
        )

    results_df = pd.DataFrame(results)

    print(f"\n{'=' * 80}")
    print("PARAMETER SWEEP RESULTS")
    print(f"{'=' * 80}")
    print(results_df.to_string(index=False))

    # Find best parameter
    best_idx = results_df["return_percent"].idxmax()
    best_n_days = results_df.loc[best_idx, "n_days_threshold"]
    best_return = results_df.loc[best_idx, "return_percent"]

    print(f"\n🏆 Best n_days_threshold: {best_n_days} (Return: {best_return:.2f}%)")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(
        results_df["n_days_threshold"], results_df["return_percent"], marker="o"
    )
    axes[0, 0].set_xlabel("n_days_threshold")
    axes[0, 0].set_ylabel("Return (%)")
    axes[0, 0].set_title("Return vs n_days_threshold")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(
        results_df["n_days_threshold"], results_df["outperformance"], marker="o"
    )
    axes[0, 1].axhline(y=0, color="r", linestyle="--")
    axes[0, 1].set_xlabel("n_days_threshold")
    axes[0, 1].set_ylabel("Outperformance (%)")
    axes[0, 1].set_title("Outperformance vs Buy & Hold")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(
        results_df["n_days_threshold"], results_df["num_trades"], marker="o"
    )
    axes[1, 0].set_xlabel("n_days_threshold")
    axes[1, 0].set_ylabel("Number of Trades")
    axes[1, 0].set_title("Trading Activity")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(
        results_df["n_days_threshold"], results_df["transaction_costs"], marker="o"
    )
    axes[1, 1].set_xlabel("n_days_threshold")
    axes[1, 1].set_ylabel("Transaction Costs ($)")
    axes[1, 1].set_title("Total Transaction Costs")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/parameter_sweep.png", dpi=150)
    print("\n✅ Parameter sweep plot saved to results/parameter_sweep.png")

    return results_df
