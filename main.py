# main.py

import os
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.data_generator import PassengerFlowGenerator
from src.models.advanced_predictor import create_predictor
from src.visualization.visualizer import create_visualizer
from src.models.fare_optimizer import create_fare_optimizer

class FareSystem:
    """Main class for the fare prediction system"""
    
    def __init__(self, config_path: str = 'configs/config.json'):
        self.config = self._load_config(config_path)
        self.setup_paths()
        self.setup_logging()
        self.visualizer = create_visualizer(self.paths['plots'])
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using default configuration")
            return {
                'simulation_params': {
                    'peak_hours': {
                        'morning': {'start': '07:00', 'end': '09:00'},
                        'evening': {'start': '16:00', 'end': '18:00'}
                    },
                    'base_fare': 39,
                    'fare_limits': {'min': 20, 'max': 60}
                }
            }
    
    def setup_paths(self):
        """Create necessary directories"""
        self.paths = {
            'data': Path('data'),
            'models': Path('models'),
            'logs': Path('logs'),
            'results': Path('results'),
            'plots': Path('plots')
        }
        
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Configure logging"""
        log_path = self.paths['logs'] / f"system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_path, rotation="500 MB")
    
    def generate_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate passenger flow data"""
        logger.info("Generating passenger flow data...")
        generator = PassengerFlowGenerator(self.config)
        df = generator.generate_dataset(start_date, end_date)
        
        data_path = self.paths['data'] / f"passenger_flow_{start_date.date()}_{end_date.date()}.csv"
        df.to_csv(data_path, index=False)
        logger.info(f"Data saved to {data_path}")
        
        return df
    
    def train_model(self, df: pd.DataFrame) -> tuple:
        """Train the prediction model"""
        logger.info("Training prediction model...")
        predictor = create_predictor()
        
        train_loader, val_loader = predictor.prepare_data(df)
        history = predictor.train(train_loader, val_loader)
        
        # Save training history
        history_path = self.paths['results'] / 'training_history.csv'
        history.to_csv(history_path, index=False)
        logger.info(f"Training history saved to {history_path}")
        
        return predictor, history
    
    def evaluate_predictions(self, df: pd.DataFrame, predictions: np.ndarray) -> dict:
        """Calculate prediction metrics"""
        metrics = {
            'MAE': mean_absolute_error(df['passengers'], predictions),
            'RMSE': np.sqrt(mean_squared_error(df['passengers'], predictions)),
            'R2': r2_score(df['passengers'], predictions)
        }
        
        logger.info("Prediction Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        return metrics
    
    # In main.py, update the run method:

    def run(self, days_to_simulate: int = 30):
        """Run the complete system"""
        try:
            # Generate data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_to_simulate)
            df = self.generate_data(start_date, end_date)
            
            # Train model and generate predictions
            predictor, history = self.train_model(df)
            predictions = predictor.predict(df)
            
            # Create fare optimizer and generate recommendations
            fare_optimizer = create_fare_optimizer(self.config.get('fare_params'))
            fare_recommendations = fare_optimizer.generate_fare_recommendations(df, predictions)
            # After generating fare recommendations
            self.visualizer.plot_fare_patterns(fare_recommendations)
            # Create visualizations
            self.visualizer.plot_training_history(history)
            self.visualizer.plot_predictions(df, predictions)
            self.visualizer.plot_station_heatmap(df)
            
            # Evaluate predictions
            metrics = self.evaluate_predictions(df, predictions)
            self.visualizer.plot_metrics_summary(metrics)
            
            # Save results
            results = fare_recommendations.copy()
            results_path = self.paths['results'] / f"fare_recommendations_{datetime.now().strftime('%Y%m%d')}.csv"
            results.to_csv(results_path, index=False)
            
            logger.info(f"Generated fare recommendations saved to {results_path}")
            logger.info("\nSample fare recommendations:")
            logger.info("\n" + str(results.head()))
            
            # Print summary statistics
            logger.info("\nFare Optimization Summary:")
            logger.info(f"Average optimal fare: {results['optimal_fare'].mean():.2f} SEK")
            logger.info(f"Peak hour average fare: {results[results['datetime'].dt.hour.isin(range(7,10))]['optimal_fare'].mean():.2f} SEK")
            logger.info(f"Off-peak average fare: {results[~results['datetime'].dt.hour.isin(range(7,10))]['optimal_fare'].mean():.2f} SEK")
            logger.info(f"Estimated daily revenue: {results.groupby(results['datetime'].dt.date)['estimated_revenue'].sum().mean():.2f} SEK")
            
            return results, metrics
            
        except Exception as e:
            logger.exception(f"Error running system: {str(e)}")
            raise

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SL Fare System')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--days', 
        type=int, 
        default=30,
        help='Number of days to simulate'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logger.level("DEBUG")
    
    try:
        # Initialize and run system
        system = FareSystem(args.config)
        results, metrics = system.run(days_to_simulate=args.days)
        logger.info("System execution completed successfully")
        
    except Exception as e:
        logger.exception(f"Error running system: {str(e)}")
        raise

if __name__ == "__main__":
    main()



# # Basic usage
# python main.py

# # With debug mode
# python main.py --debug

# # Specify simulation days
# python main.py --days 30

# # Custom config file
# python main.py --config path/to/config.json

#pip install seaborn --trusted-host pypi.org --trusted-host files.pythonhosted.org