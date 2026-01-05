import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass

@dataclass
class TradingConfig:
    """Configuration for enhanced trading strategy"""
    # Signal thresholds
    min_signal_threshold: float = 0.001  # Minimum predicted return to trade (0.1%)
    max_volatility_threshold: float = 0.05  # Max volatility (5%)
    cooldown_period: int = 5  # Minimum periods between trades
    
    # Position sizing
    base_position_size: float = 0.1  # Base position size (10% of portfolio)
    max_position_size: float = 0.5   # Maximum position size (50%)
    position_scaling: bool = True    # Scale position by signal strength
    
    # Risk management
    max_drawdown: float = 0.2        # Maximum drawdown (20%)
    stop_loss: float = 0.02          # Stop loss (2%)
    take_profit: float = 0.04        # Take profit (4%)
    
    # Transaction costs
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005         # 0.05% slippage
    
    # Strategy options
    long_only: bool = False           # Only long positions
    use_volatility_filter: bool = True
    use_signal_filter: bool = True
    use_cooldown: bool = True

class EnhancedBacktester:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset backtester state"""
        self.portfolio_value = []
        self.positions = []
        self.trades = []
        self.drawdowns = []
        self.returns = []
        self.current_position = 0.0
        self.cash = 1.0  # Start with 1 unit of cash
        self.last_trade_idx = -self.config.cooldown_period
        self.peak_value = 1.0
    
    def calculate_position_size(self, signal: float, volatility: float) -> float:
        """Calculate position size based on signal strength and volatility"""
        if not self.config.position_scaling:
            return self.config.base_position_size
        
        # Scale position by signal strength (capped)
        signal_strength = min(abs(signal) / self.config.min_signal_threshold, 2.0)
        
        # Reduce position size in high volatility
        volatility_adjustment = 1.0 - min(volatility / self.config.max_volatility_threshold, 0.5)
        
        position_size = self.config.base_position_size * signal_strength * volatility_adjustment
        return min(position_size, self.config.max_position_size)
    
    def should_trade(self, signal: float, volatility: float, idx: int) -> bool:
        """Determine if we should place a trade"""
        # Signal filter
        if self.config.use_signal_filter and abs(signal) < self.config.min_signal_threshold:
            return False
        
        # Volatility filter
        if self.config.use_volatility_filter and volatility > self.config.max_volatility_threshold:
            return False
        
        # Cooldown filter
        if self.config.use_cooldown and idx - self.last_trade_idx < self.config.cooldown_period:
            return False
        
        # Long-only filter
        if self.config.long_only and signal < 0:
            return False
        
        return True
    
    def execute_trade(self, signal: float, price: float, volatility: float, idx: int):
        """Execute a trade based on signal"""
        if not self.should_trade(signal, volatility, idx):
            return
        
        # Calculate position size
        position_size = self.calculate_position_size(signal, volatility)
        
        # Determine trade direction
        if signal > 0:
            # Long position
            if self.current_position < 0:
                # Close short position first
                self.close_position(price, idx)
            
            # Open or add to long position
            trade_value = self.cash * position_size
            shares = trade_value / (price * (1 + self.config.slippage))
            cost = trade_value * self.config.transaction_cost
            
            self.current_position += shares
            self.cash -= (trade_value + cost)
            
            self.trades.append({
                'idx': idx,
                'type': 'buy',
                'price': price,
                'shares': shares,
                'value': trade_value,
                'cost': cost,
                'signal': signal
            })
            
        elif signal < 0 and not self.config.long_only:
            # Short position
            if self.current_position > 0:
                # Close long position first
                self.close_position(price, idx)
            
            # Open short position
            trade_value = self.cash * position_size
            shares = trade_value / (price * (1 - self.config.slippage))
            cost = trade_value * self.config.transaction_cost
            
            self.current_position -= shares
            self.cash -= cost
            
            self.trades.append({
                'idx': idx,
                'type': 'sell_short',
                'price': price,
                'shares': shares,
                'value': trade_value,
                'cost': cost,
                'signal': signal
            })
        
        self.last_trade_idx = idx
    
    def close_position(self, price: float, idx: int):
        """Close current position"""
        if self.current_position == 0:
            return
        
        if self.current_position > 0:
            # Close long position
            proceeds = self.current_position * price * (1 - self.config.slippage)
            cost = proceeds * self.config.transaction_cost
            self.cash += (proceeds - cost)
            
            self.trades.append({
                'idx': idx,
                'type': 'sell',
                'price': price,
                'shares': self.current_position,
                'value': proceeds,
                'cost': cost,
                'signal': 0
            })
        else:
            # Close short position
            proceeds = abs(self.current_position) * price * (1 + self.config.slippage)
            cost = proceeds * self.config.transaction_cost
            self.cash += (proceeds - cost)
            
            self.trades.append({
                'idx': idx,
                'type': 'buy_to_cover',
                'price': price,
                'shares': abs(self.current_position),
                'value': proceeds,
                'cost': cost,
                'signal': 0
            })
        
        self.current_position = 0
    
    def check_risk_limits(self, price: float, idx: int):
        """Check and enforce risk limits"""
        current_value = self.cash + self.current_position * price
        
        # Check max drawdown
        if current_value < self.peak_value * (1 - self.config.max_drawdown):
            self.close_position(price, idx)
            return True  # Emergency exit
        
        # Update peak
        self.peak_value = max(self.peak_value, current_value)
        return False
    
    def backtest(self, predictions: np.ndarray, prices: np.ndarray, 
                 volatilities: np.ndarray, timestamps: List[str]) -> Dict:
        """
        Run backtest with predictions
        
        Args:
            predictions: Model predictions (returns or class probabilities)
            prices: Actual prices
            volatilities: Volatility estimates
            timestamps: List of timestamps
        """
        self.reset()
        
        for i in range(len(predictions)):
            price = prices[i]
            volatility = volatilities[i]
            signal = predictions[i]
            
            # Check risk limits
            emergency_exit = self.check_risk_limits(price, i)
            
            # Execute trades
            if not emergency_exit:
                self.execute_trade(signal, price, volatility, i)
            
            # Calculate portfolio value
            portfolio_value = self.cash + self.current_position * price
            self.portfolio_value.append(portfolio_value)
            self.positions.append(self.current_position)
            
            # Calculate returns
            if i > 0:
                ret = (portfolio_value - self.portfolio_value[-2]) / self.portfolio_value[-2]
                self.returns.append(ret)
        
        # Close final position
        if len(prices) > 0:
            self.close_position(prices[-1], len(prices) - 1)
            final_value = self.cash
            self.portfolio_value.append(final_value)
        
        return self.calculate_metrics(timestamps)
    
    def calculate_metrics(self, timestamps: List[str]) -> Dict:
        """Calculate performance metrics"""
        portfolio_values = np.array(self.portfolio_value)
        returns = np.array(self.returns)
        
        # Basic metrics
        total_return = (portfolio_values[-1] - 1.0) / 1.0
        annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Trade statistics
        num_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['type'] in ['sell', 'buy_to_cover'] and t['value'] > 0]
        win_rate = len(winning_trades) / max(num_trades, 1)
        
        # Buy and hold comparison
        if len(portfolio_values) > 1:
            buy_hold_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        else:
            buy_hold_return = 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'buy_hold_return': buy_hold_return,
            'final_value': portfolio_values[-1] if len(portfolio_values) > 0 else 1.0,
            'portfolio_values': portfolio_values.tolist(),
            'positions': self.positions,
            'trades': self.trades,
            'timestamps': timestamps
        }
        
        return metrics
    
    def plot_results(self, metrics: Dict, save_path: str = None):
        """Plot backtest results"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Portfolio value
        axes[0].plot(metrics['portfolio_values'], label='Strategy', linewidth=2)
        axes[0].set_title('Portfolio Value')
        axes[0].set_ylabel('Portfolio Value')
        axes[0].legend()
        axes[0].grid(True)
        
        # Drawdown
        portfolio_values = np.array(metrics['portfolio_values'])
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        axes[1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[1].plot(drawdown, color='red', linewidth=1)
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown')
        axes[1].grid(True)
        
        # Positions
        axes[2].plot(metrics['positions'], label='Position Size', linewidth=1)
        axes[2].set_title('Position Size')
        axes[2].set_ylabel('Shares')
        axes[2].set_xlabel('Time')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, metrics: Dict, save_path: str):
        """Save backtest results to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, (np.int64, np.float64)):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

def main():
    """Example usage"""
    # Configuration
    config = TradingConfig()
    config.min_signal_threshold = 0.002
    config.max_volatility_threshold = 0.03
    config.position_scaling = True
    
    # Load data and predictions (example)
    # In practice, you would load your actual data and model predictions
    data = pd.read_csv('../../data/processed/btcusdt_enriched.csv')
    
    # Generate dummy predictions for demonstration
    predictions = np.random.normal(0, 0.01, len(data))
    prices = data['close'].values
    volatilities = data['volatility_5'].fillna(0.02).values
    timestamps = data['timestamp'].tolist()
    
    # Run backtest
    backtester = EnhancedBacktester(config)
    metrics = backtester.backtest(predictions, prices, volatilities, timestamps)
    
    # Print results
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    
    # Plot and save results
    backtester.plot_results(metrics, 'backtest_results.png')
    backtester.save_results(metrics, 'backtest_results.json')

if __name__ == "__main__":
    main()
