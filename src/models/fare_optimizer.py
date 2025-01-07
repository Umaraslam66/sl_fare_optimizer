# src/models/fare_optimizer.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass
from loguru import logger

@dataclass
class FareConfig:
    """Configuration for fare optimization"""
    base_fare: float = 39.0  # Base fare in SEK
    min_fare: float = 20.0   # Minimum allowed fare
    max_fare: float = 60.0   # Maximum allowed fare
    peak_multiplier: float = 1.5  # Maximum peak hour multiplier
    off_peak_discount: float = 0.8  # Off-peak discount
    demand_sensitivity: float = -0.3  # Price elasticity of demand
    capacity_threshold: float = 0.8  # Threshold for capacity-based pricing

class FareOptimizer:
    """Optimize fares based on predicted demand and time of day"""
    
    def __init__(self, config: FareConfig = None):
        self.config = config or FareConfig()
    
    def is_peak_hour(self, hour: int) -> bool:
        """Determine if given hour is a peak hour"""
        morning_peak = (7 <= hour <= 9)
        evening_peak = (16 <= hour <= 18)
        return morning_peak or evening_peak
    
    def calculate_demand_multiplier(self, current_demand: float, max_capacity: float) -> float:
        """Calculate price multiplier based on demand vs capacity"""
        capacity_utilization = current_demand / max_capacity
        if capacity_utilization > self.config.capacity_threshold:
            # Increase price as demand approaches capacity
            return 1.0 + (capacity_utilization - self.config.capacity_threshold) / (1 - self.config.capacity_threshold)
        return 1.0
    
    def estimate_revenue(self, demand: float, fare: float) -> float:
        """Estimate revenue for a given demand and fare"""
        # Apply price elasticity to adjust demand
        demand_multiplier = (fare / self.config.base_fare) ** self.config.demand_sensitivity
        adjusted_demand = demand * demand_multiplier
        return adjusted_demand * fare
    
    # src/models/fare_optimizer.py

    def find_optimal_fare(self, demand: float, max_capacity: float, is_peak: bool) -> Tuple[float, Dict]:
        """Find optimal fare that maximizes revenue while considering constraints"""
        best_fare = self.config.base_fare
        best_revenue = 0
        best_details = {
            'base_fare': self.config.base_fare,
            'time_multiplier': 1.0,
            'demand_multiplier': 1.0,
            'estimated_revenue': 0.0,
            'adjusted_demand': demand
        }
        
        # Test range of fares
        test_fares = np.linspace(self.config.min_fare, self.config.max_fare, 50)
        for fare in test_fares:
            # Apply time-based adjustment
            time_multiplier = self.config.peak_multiplier if is_peak else self.config.off_peak_discount
            
            # Apply demand-based adjustment
            demand_multiplier = self.calculate_demand_multiplier(demand, max_capacity)
            
            # Calculate adjusted fare
            adjusted_fare = fare * time_multiplier * demand_multiplier
            adjusted_fare = min(max(adjusted_fare, self.config.min_fare), self.config.max_fare)
            
            # Calculate revenue
            revenue = self.estimate_revenue(demand, adjusted_fare)
            
            if revenue > best_revenue:
                best_revenue = revenue
                best_fare = adjusted_fare
                best_details = {
                    'base_fare': fare,
                    'time_multiplier': time_multiplier,
                    'demand_multiplier': demand_multiplier,
                    'estimated_revenue': revenue,
                    'adjusted_demand': demand * (adjusted_fare / self.config.base_fare) ** self.config.demand_sensitivity
                }
        
        return best_fare, best_details

    def optimize_fares(self, df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
        """Generate optimized fares for each station and time period"""
        results = df.copy()
        optimized_fares = []
        revenue_estimates = []
        demand_multipliers = []
        time_multipliers = []
        adjusted_demands = []
        
        # Calculate max capacity for each station
        station_capacities = df.groupby('station')['passengers'].quantile(0.95)
        
        for idx, row in df.iterrows():
            station = row['station']
            hour = row['datetime'].hour
            predicted_demand = predictions[idx]
            max_capacity = station_capacities[station]
            
            # Find optimal fare
            optimal_fare, details = self.find_optimal_fare(
                demand=predicted_demand,
                max_capacity=max_capacity,
                is_peak=self.is_peak_hour(hour)
            )
            
            optimized_fares.append(optimal_fare)
            revenue_estimates.append(details['estimated_revenue'])
            demand_multipliers.append(details['demand_multiplier'])
            time_multipliers.append(details['time_multiplier'])
            adjusted_demands.append(details['adjusted_demand'])
        
        # Add results to dataframe
        results['optimal_fare'] = optimized_fares
        results['estimated_revenue'] = revenue_estimates
        results['demand_multiplier'] = demand_multipliers
        results['time_multiplier'] = time_multipliers
        results['predicted_demand'] = predictions
        results['adjusted_demand'] = adjusted_demands
        
        return results

    def generate_fare_recommendations(self, df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
        """Generate fare recommendations with explanations"""
        optimized_results = self.optimize_fares(df, predictions)
        
        # Add recommendations and explanations
        recommendations = []
        for idx, row in optimized_results.iterrows():
            if self.is_peak_hour(row['datetime'].hour):
                if row['optimal_fare'] > self.config.base_fare:
                    reason = "Peak hour with high demand"
                else:
                    reason = "Peak hour but lower than expected demand"
            else:
                if row['optimal_fare'] < self.config.base_fare:
                    reason = "Off-peak discount applied"
                else:
                    reason = "Off-peak but high demand detected"
            
            recommendations.append({
                'datetime': row['datetime'],
                'station': row['station'],
                'current_demand': row['passengers'],
                'predicted_demand': row['predicted_demand'],
                'optimal_fare': row['optimal_fare'],
                'estimated_revenue': row['estimated_revenue'],
                'demand_multiplier': row['demand_multiplier'],
                'time_multiplier': row['time_multiplier'],
                'adjusted_demand': row['adjusted_demand'],
                'recommendation_reason': reason
            })
        
        return pd.DataFrame(recommendations)

    # Add a method to get summary statistics
    def get_summary_stats(self, recommendations: pd.DataFrame) -> Dict:
        """Get summary statistics for fare recommendations"""
        return {
            'avg_optimal_fare': recommendations['optimal_fare'].mean(),
            'peak_avg_fare': recommendations[recommendations['datetime'].dt.hour.isin(range(7,10))]['optimal_fare'].mean(),
            'off_peak_avg_fare': recommendations[~recommendations['datetime'].dt.hour.isin(range(7,10))]['optimal_fare'].mean(),
            'total_estimated_revenue': recommendations['estimated_revenue'].sum(),
            'avg_demand_multiplier': recommendations['demand_multiplier'].mean(),
            'avg_time_multiplier': recommendations['time_multiplier'].mean()
        }

def create_fare_optimizer(config: Dict = None) -> FareOptimizer:
    """Create a fare optimizer instance with optional config"""
    if config:
        fare_config = FareConfig(
            base_fare=config.get('base_fare', 39.0),
            min_fare=config.get('min_fare', 20.0),
            max_fare=config.get('max_fare', 60.0),
            peak_multiplier=config.get('peak_multiplier', 1.5),
            off_peak_discount=config.get('off_peak_discount', 0.8),
            demand_sensitivity=config.get('demand_sensitivity', -0.3),
            capacity_threshold=config.get('capacity_threshold', 0.8)
        )
    else:
        fare_config = FareConfig()
    
    return FareOptimizer(fare_config)