# src/visualization/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime

class Visualizer:
    """Visualization tools for the fare prediction system"""
    
    def __init__(self, save_dir: str = 'plots'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")  # Use seaborn's set_style instead of plt.style
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Configure plotting style"""
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 12
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
        
    def plot_training_history(
        self,
        history: pd.DataFrame,
        filename: str = 'training_history.png'
    ) -> str:
        """Plot training history"""
        plt.figure()
        
        sns.lineplot(data=history, x='epoch', y='train_loss', label='Train Loss')
        sns.lineplot(data=history, x='epoch', y='val_loss', label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        save_path = self.save_dir / filename
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(save_path)


    def plot_predictions(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        station: Optional[str] = None,
        filename: str = 'predictions.png'
    ) -> str:
        """Plot actual vs predicted values"""
        plt.figure(figsize=(15, 8))
        
        if station is None:
            station = df['station'].unique()[0]
        
        # Get station data
        station_mask = df['station'] == station
        station_data = df[station_mask].copy()
        station_data = station_data.reset_index(drop=True)
        
        # Get predictions for this station
        station_indices = np.where(station_mask)[0]
        station_predictions = predictions[station_indices]
        
        # Plot actual and predicted values
        plt.plot(station_data['datetime'], station_data['passengers'],
                label='Actual', alpha=0.7, linewidth=2)
        plt.plot(station_data['datetime'], station_predictions,
                label='Predicted', alpha=0.7, linewidth=2, linestyle='--')
        
        plt.xlabel('Time')
        plt.ylabel('Number of Passengers')
        plt.title(f'Passenger Flow Predictions for {station}')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Format y-axis with thousands separator
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: format(int(x), ','))
        )
        
        save_path = self.save_dir / filename
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(save_path)

    def plot_station_heatmap(
        self,
        df: pd.DataFrame,
        filename: str = 'station_heatmap.png'
    ) -> str:
        """Create heatmap of passenger flow by station and hour"""
        plt.figure(figsize=(15, 10))
        
        # Create pivot table for heatmap
        pivot_df = df.pivot_table(
            index='station',
            columns=df['datetime'].dt.hour,
            values='passengers',
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(
            pivot_df,
            cmap='viridis',
            center=pivot_df.mean().mean(),
            annot=True,
            fmt='.0f',
            cbar_kws={'label': 'Average Passengers'}
        )
        
        plt.title('Average Passenger Flow by Station and Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Station')
        
        # Rotate x-axis labels
        plt.xticks(rotation=0)
        
        save_path = self.save_dir / filename
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(save_path)

    def plot_metrics_summary(
        self,
        metrics: Dict,
        filename: str = 'metrics_summary.png'
    ) -> str:
        """Plot summary of performance metrics"""
        plt.figure(figsize=(10, 6))
        
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        
        # Create bar plot
        sns.barplot(data=metrics_df, x='Metric', y='Value')
        
        plt.title('Model Performance Metrics')
        plt.xticks(rotation=45)
        
        # Add value labels on top of bars
        for i, v in enumerate(metrics.values()):
            plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        save_path = self.save_dir / filename
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(save_path)

    def plot_training_history(
        self,
        history: pd.DataFrame,
        filename: str = 'training_history.png'
    ) -> str:
        """Plot training history"""
        plt.figure()
        
        # Plot both training and validation loss
        sns.lineplot(data=history, x='epoch', y='train_loss', label='Train Loss')
        sns.lineplot(data=history, x='epoch', y='val_loss', label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = self.save_dir / filename
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(save_path)
    # Add to src/visualization/visualizer.py

    def plot_fare_patterns(
        self,
        results: pd.DataFrame,
        station: Optional[str] = None,
        filename: str = 'fare_patterns.png'
    ) -> str:
        """Plot fare patterns with demand"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])
        
        if station is None:
            station = results['station'].unique()[0]
        
        station_data = results[results['station'] == station].copy()
        
        # Plot 1: Fares and Demand
        ax1.plot(station_data['datetime'], station_data['optimal_fare'], 
                label='Optimal Fare', color='blue')
        ax1.axhline(y=39, color='gray', linestyle='--', label='Base Fare')
        
        # Add demand on secondary axis
        ax2_twin = ax1.twinx()
        ax2_twin.plot(station_data['datetime'], station_data['predicted_demand'], 
                    label='Predicted Demand', color='red', alpha=0.5)
        
        # Formatting
        ax1.set_title(f'Fare Optimization Patterns for {station}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Fare (SEK)')
        ax2_twin.set_ylabel('Predicted Passengers')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot 2: Daily Pattern
        station_data['hour'] = station_data['datetime'].dt.hour
        hourly_fares = station_data.groupby('hour')['optimal_fare'].mean()
        
        ax2.plot(hourly_fares.index, hourly_fares.values, marker='o')
        ax2.set_title('Average Daily Fare Pattern')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average Fare (SEK)')
        ax2.grid(True, alpha=0.3)
        
        # Highlight peak hours
        peak_hours_morning = np.array([7, 8, 9])
        peak_hours_evening = np.array([16, 17, 18])
        ax2.fill_between(peak_hours_morning, 0, max(hourly_fares.values), 
                        alpha=0.2, color='yellow', label='Peak Hours')
        ax2.fill_between(peak_hours_evening, 0, max(hourly_fares.values), 
                        alpha=0.2, color='yellow')
        
        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(save_path)

def create_visualizer(save_dir: str = 'plots') -> Visualizer:
    """Create a visualizer instance"""
    return Visualizer(save_dir)