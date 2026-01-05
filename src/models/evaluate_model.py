import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, confusion_matrix
from typing import Dict, List, Tuple, Any
import seaborn as sns
from pathlib import Path

from .enhanced_backtest import EnhancedBacktester, TradingConfig
from .transformer_model import TradingTransformer

class ModelEvaluator:
    def __init__(self, model_path: str, config_path: str = None):
        """Initialize evaluator with trained model"""
        self.model_path = model_path
        self.model = None
        self.config = None
        self.scaler = None
        self.mode = None
        
        self.load_model()
    
    def load_model(self):
        """Load trained model and configuration"""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        self.config = checkpoint.get('config')
        self.scaler = checkpoint.get('scaler')
        self.mode = checkpoint.get('mode', 'regression')
        
        # Reconstruct model
        self.model = TradingTransformer(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def prepare_data(self, data_path: str, sequence_length: int = 16):
        """Prepare data for evaluation"""
        df = pd.read_csv(data_path)
        
        # Select feature columns (exclude timestamp and label)
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'label']]
        features = df[feature_cols].values
        labels = df['label'].values
        prices = df['close'].values
        timestamps = df['timestamp'].tolist()
        
        # Scale features
        features = self.scaler.transform(features)
        
        # Create sequences
        X, y, price_seq, ts_seq = [], [], [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(labels[i + sequence_length])
            price_seq.append(prices[i + sequence_length])
            ts_seq.append(timestamps[i + sequence_length])
        
        return np.array(X), np.array(y), np.array(price_seq), ts_seq
    
    def predict(self, X: np.ndarray, batch_size: int = 64):
        """Make predictions on data"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i:i+batch_size])
                batch_preds = self.model(batch)
                predictions.extend(batch_preds.cpu().numpy())
        
        return np.array(predictions)
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Calculate directional accuracy
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        directional_accuracy = np.mean(true_direction == pred_direction)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'directional_accuracy': directional_accuracy
        }
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate classification metrics"""
        if self.mode == '3class':
            # Convert from [-1, 0, 1] to [0, 1, 2] for sklearn
            y_true_adj = (y_true + 1).astype(int)
            y_pred_adj = (y_pred + 1).astype(int)
            
            report = classification_report(y_true_adj, y_pred_adj, output_dict=True)
            cm = confusion_matrix(y_true_adj, y_pred_adj)
            
            return {
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'accuracy': report['accuracy']
            }
        else:  # binary
            y_pred_binary = (y_pred > 0.5).astype(int)
            report = classification_report(y_true, y_pred_binary, output_dict=True)
            cm = confusion_matrix(y_true, y_pred_binary)
            
            return {
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'accuracy': report['accuracy']
            }
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            prices: np.ndarray, timestamps: List[str]) -> Dict:
        """Comprehensive evaluation of predictions"""
        metrics = {}
        
        # Task-specific metrics
        if self.mode == 'regression':
            metrics.update(self.calculate_regression_metrics(y_true, y_pred))
        else:
            metrics.update(self.calculate_classification_metrics(y_true, y_pred))
        
        # Backtesting metrics
        # Calculate volatilities for backtesting
        returns = np.diff(np.log(prices))
        volatilities = pd.Series(returns).rolling(20).std().fillna(0.02).values
        
        # Align volatilities with predictions
        if len(volatilities) < len(y_pred):
            volatilities = np.pad(volatilities, (len(y_pred) - len(volatilities), 0), 'constant')
        else:
            volatilities = volatilities[:len(y_pred)]
        
        # Run backtest with different strategies
        strategies = {
            'conservative': TradingConfig(
                min_signal_threshold=0.005,
                max_volatility_threshold=0.02,
                position_scaling=True
            ),
            'aggressive': TradingConfig(
                min_signal_threshold=0.001,
                max_volatility_threshold=0.05,
                position_scaling=True
            ),
            'balanced': TradingConfig(
                min_signal_threshold=0.002,
                max_volatility_threshold=0.03,
                position_scaling=True
            )
        }
        
        backtest_results = {}
        for strategy_name, config in strategies.items():
            backtester = EnhancedBacktester(config)
            backtest_metrics = backtester.backtest(y_pred, prices, volatilities, timestamps)
            
            # Extract key metrics
            backtest_results[strategy_name] = {
                'total_return': backtest_metrics['total_return'],
                'sharpe_ratio': backtest_metrics['sharpe_ratio'],
                'max_drawdown': backtest_metrics['max_drawdown'],
                'win_rate': backtest_metrics['win_rate'],
                'num_trades': backtest_metrics['num_trades'],
                'buy_hold_return': backtest_metrics['buy_hold_return']
            }
        
        metrics['backtest_results'] = backtest_results
        
        # Risk-adjusted metrics
        best_strategy = max(backtest_results.keys(), 
                          key=lambda k: backtest_results[k]['sharpe_ratio'])
        metrics['best_strategy'] = best_strategy
        metrics['best_sharpe'] = backtest_results[best_strategy]['sharpe_ratio']
        metrics['best_return'] = backtest_results[best_strategy]['total_return']
        
        return metrics
    
    def plot_evaluation(self, metrics: Dict, y_true: np.ndarray, y_pred: np.ndarray,
                       save_dir: str = 'evaluation_results'):
        """Create evaluation plots"""
        Path(save_dir).mkdir(exist_ok=True)
        
        # Plot 1: Predictions vs Actual
        plt.figure(figsize=(12, 8))
        
        if self.mode == 'regression':
            plt.subplot(2, 2, 1)
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            plt.xlabel('Actual Returns')
            plt.ylabel('Predicted Returns')
            plt.title('Predictions vs Actual')
            
            plt.subplot(2, 2, 2)
            plt.plot(y_true[:100], label='Actual', alpha=0.7)
            plt.plot(y_pred[:100], label='Predicted', alpha=0.7)
            plt.title('Returns Time Series (First 100)')
            plt.legend()
        else:
            # Classification plots
            plt.subplot(2, 2, 1)
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            
            plt.subplot(2, 2, 2)
            if self.mode == '3class':
                classes = ['Down', 'Neutral', 'Up']
            else:
                classes = ['Down', 'Up']
            
            report = metrics['classification_report']
            accuracies = [report[str(i)]['precision'] for i in range(len(classes))]
            plt.bar(classes, accuracies)
            plt.title('Precision by Class')
            plt.ylabel('Precision')
        
        # Plot 3: Backtest Comparison
        plt.subplot(2, 2, 3)
        strategies = list(metrics['backtest_results'].keys())
        returns = [metrics['backtest_results'][s]['total_return'] for s in strategies]
        sharpe_ratios = [metrics['backtest_results'][s]['sharpe_ratio'] for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        plt.bar(x - width/2, returns, width, label='Total Return', alpha=0.7)
        plt.bar(x + width/2, sharpe_ratios, width, label='Sharpe Ratio', alpha=0.7)
        plt.xlabel('Strategy')
        plt.ylabel('Value')
        plt.title('Strategy Performance')
        plt.xticks(x, strategies)
        plt.legend()
        
        # Plot 4: Risk Metrics
        plt.subplot(2, 2, 4)
        drawdowns = [metrics['backtest_results'][s]['max_drawdown'] for s in strategies]
        plt.bar(strategies, np.abs(drawdowns), color='red', alpha=0.7)
        plt.title('Maximum Drawdown')
        plt.ylabel('Drawdown (absolute value)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_evaluation_report(self, metrics: Dict, save_path: str):
        """Save comprehensive evaluation report"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, (np.int64, np.float64)):
                serializable_metrics[key] = float(value)
            elif isinstance(value, dict):
                # Handle nested dictionaries
                nested_dict = {}
                for k, v in value.items():
                    if isinstance(v, (np.int64, np.float64)):
                        nested_dict[k] = float(v)
                    else:
                        nested_dict[k] = v
                serializable_metrics[key] = nested_dict
            else:
                serializable_metrics[key] = value
        
        # Add summary
        summary = {
            'model_path': self.model_path,
            'mode': self.mode,
            'best_strategy': metrics.get('best_strategy', 'N/A'),
            'best_sharpe_ratio': metrics.get('best_sharpe', 0),
            'best_total_return': metrics.get('best_return', 0),
            'evaluation_date': pd.Timestamp.now().isoformat()
        }
        
        serializable_metrics['summary'] = summary
        
        with open(save_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def evaluate(self, data_path: str, save_dir: str = 'evaluation_results'):
        """Run complete evaluation"""
        print(f"Evaluating model in {self.mode} mode...")
        
        # Prepare data
        X, y, prices, timestamps = self.prepare_data(data_path)
        
        # Make predictions
        predictions = self.predict(X)
        
        # Evaluate predictions
        metrics = self.evaluate_predictions(y, predictions, prices, timestamps)
        
        # Print summary
        print("\n=== Evaluation Summary ===")
        print(f"Mode: {self.mode}")
        print(f"Best Strategy: {metrics.get('best_strategy', 'N/A')}")
        print(f"Best Sharpe Ratio: {metrics.get('best_sharpe', 0):.3f}")
        print(f"Best Total Return: {metrics.get('best_return', 0):.2%}")
        
        if self.mode == 'regression':
            print(f"RMSE: {metrics.get('rmse', 0):.4f}")
            print(f"Directional Accuracy: {metrics.get('directional_accuracy', 0):.2%}")
        else:
            print(f"Accuracy: {metrics.get('accuracy', 0):.2%}")
        
        print("\n=== Strategy Performance ===")
        for strategy, results in metrics['backtest_results'].items():
            print(f"{strategy}: Return={results['total_return']:.2%}, "
                  f"Sharpe={results['sharpe_ratio']:.2f}, "
                  f"Drawdown={results['max_drawdown']:.2%}")
        
        # Save results
        Path(save_dir).mkdir(exist_ok=True)
        self.plot_evaluation(metrics, y, predictions, save_dir)
        self.save_evaluation_report(metrics, f'{save_dir}/evaluation_report.json')
        
        print(f"\nResults saved to {save_dir}/")
        return metrics

def main():
    """Example usage"""
    # Evaluate regression model
    evaluator = ModelEvaluator('models/enhanced_transformer/best_regression_model.pt')
    metrics = evaluator.evaluate('../../data/processed/btcusdt_enriched.csv')
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
