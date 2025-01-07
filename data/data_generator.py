# src/data/data_generator.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

class StationData:
    """Manages station-specific data and characteristics"""
    
    def __init__(self):
        # Real Stockholm metro stations with approximate daily passengers
        self.stations = {
            'T-Centralen': {'daily_avg': 300000, 'type': 'hub', 'lines': ['red', 'green', 'blue']},
            'Slussen': {'daily_avg': 150000, 'type': 'hub', 'lines': ['red', 'green']},
            'Gullmarsplan': {'daily_avg': 84000, 'type': 'hub', 'lines': ['green']},
            'Fridhemsplan': {'daily_avg': 75000, 'type': 'hub', 'lines': ['blue', 'green']},
            'Tekniska högskolan': {'daily_avg': 45000, 'type': 'university', 'lines': ['red']},
            'Universitet': {'daily_avg': 42000, 'type': 'university', 'lines': ['red']},
            'Gamla stan': {'daily_avg': 65000, 'type': 'tourist', 'lines': ['red', 'green']},
            'Kungsträdgården': {'daily_avg': 55000, 'type': 'tourist', 'lines': ['blue']},
            'Odenplan': {'daily_avg': 70000, 'type': 'residential', 'lines': ['green']},
            'Medborgarplatsen': {'daily_avg': 68000, 'type': 'residential', 'lines': ['green']}
        }
        
    def get_station_weight(self, station: str, hour: int, is_weekend: bool) -> float:
        """Calculate station-specific weight based on time and day"""
        base_weight = self.stations[station]['daily_avg'] / 300000  # Normalize by T-Centralen
        
        # Apply time-based modifications
        if self.stations[station]['type'] == 'tourist':
            if is_weekend:
                base_weight *= 1.4
            if 10 <= hour <= 18:
                base_weight *= 1.3
        elif self.stations[station]['type'] == 'university':
            if is_weekend:
                base_weight *= 0.3
            if 8 <= hour <= 16:
                base_weight *= 1.5
        elif self.stations[station]['type'] == 'residential':
            if 7 <= hour <= 9:
                base_weight *= 1.6
            elif 16 <= hour <= 18:
                base_weight *= 1.4
                
        return base_weight

class WeatherGenerator:
    """Generates realistic weather data for Stockholm"""
    
    def __init__(self):
        # Monthly average temperatures in Stockholm
        self.temp_means = {
            1: -1.6, 2: -1.8, 3: 1.1, 4: 5.8, 
            5: 11.6, 6: 15.6, 7: 17.2, 8: 16.2,
            9: 11.9, 10: 7.5, 11: 3.2, 12: -0.3
        }
        
        # Monthly precipitation probability
        self.precip_prob = {
            1: 0.4, 2: 0.3, 3: 0.3, 4: 0.3,
            5: 0.3, 6: 0.4, 7: 0.4, 8: 0.4,
            9: 0.4, 10: 0.4, 11: 0.4, 12: 0.4
        }
        
    def generate_daily_weather(self, date: datetime) -> Dict:
        """Generate weather data for a specific date"""
        month = date.month
        
        # Temperature with random variation
        temp = np.random.normal(self.temp_means[month], 3)
        
        # Precipitation
        is_precipitation = np.random.random() < self.precip_prob[month]
        
        # Weather type
        if is_precipitation:
            if temp < 0:
                weather_type = 'snow'
            else:
                weather_type = 'rain'
        else:
            weather_type = 'clear'
            
        return {
            'temperature': round(temp, 1),
            'weather_type': weather_type,
            'is_precipitation': is_precipitation
        }

class EventGenerator:
    """Generates special events that affect passenger flow"""
    
    def __init__(self):
        self.annual_events = {
            (7, 1): {'name': 'Stockholm Pride', 'duration': 6, 'impact': 1.4},
            (12, 24): {'name': 'Christmas Eve', 'duration': 1, 'impact': 0.4},
            (12, 31): {'name': 'New Year\'s Eve', 'duration': 1, 'impact': 1.5},
            (6, 6): {'name': 'National Day', 'duration': 1, 'impact': 1.2}
        }
        
        self.random_events = [
            {'name': 'Concert', 'impact': 1.3, 'duration': 1},
            {'name': 'Sports Match', 'impact': 1.25, 'duration': 1},
            {'name': 'Festival', 'impact': 1.35, 'duration': 3},
            {'name': 'Conference', 'impact': 1.2, 'duration': 2}
        ]
    
    def get_events(self, date: datetime) -> List[Dict]:
        """Get all events for a specific date"""
        events = []
        
        # Check annual events
        month_day = (date.month, date.day)
        if month_day in self.annual_events:
            events.append(self.annual_events[month_day])
        
        # Random events (10% chance per day)
        if np.random.random() < 0.1:
            event = np.random.choice(self.random_events)
            events.append(event)
            
        return events

class PassengerFlowGenerator:
    """Main class for generating passenger flow data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.stations = StationData()
        self.weather_gen = WeatherGenerator()
        self.event_gen = EventGenerator()
        
    def generate_base_flow(self, hour: int, is_weekend: bool) -> float:
        """Generate base passenger flow for a given hour"""
        if is_weekend:
            base = 0.4  # 40% of weekday traffic
            if 10 <= hour <= 18:
                base *= 1.5  # Weekend afternoon peak
        else:
            base = 1.0
            if 7 <= hour <= 9:  # Morning peak
                base *= 2.0
            elif 16 <= hour <= 18:  # Evening peak
                base *= 1.8
            elif 0 <= hour <= 5:  # Night hours
                base *= 0.1
                
        return base
    
    def apply_weather_impact(self, base_flow: float, weather: Dict) -> float:
        """Modify passenger flow based on weather conditions"""
        if weather['weather_type'] == 'snow':
            base_flow *= 0.8
        elif weather['weather_type'] == 'rain':
            base_flow *= 0.9
        
        # Temperature impact
        if weather['temperature'] < -10:
            base_flow *= 0.85
        elif weather['temperature'] > 25:
            base_flow *= 0.95
            
        return base_flow
    
    def generate_daily_data(self, date: datetime) -> pd.DataFrame:
        """Generate passenger flow data for a single day"""
        is_weekend = date.weekday() >= 5
        weather = self.weather_gen.generate_daily_weather(date)
        events = self.event_gen.get_events(date)
        
        data = []
        
        for hour in range(24):
            base_flow = self.generate_base_flow(hour, is_weekend)
            weather_adjusted_flow = self.apply_weather_impact(base_flow, weather)
            
            # Apply event impacts
            event_multiplier = 1.0
            for event in events:
                event_multiplier *= event['impact']
            
            final_flow = weather_adjusted_flow * event_multiplier
            
            # Generate data for each station
            for station in self.stations.stations:
                station_weight = self.stations.get_station_weight(station, hour, is_weekend)
                passengers = int(final_flow * station_weight * 1000)  # Scale to actual numbers
                
                data.append({
                    'datetime': date + timedelta(hours=hour),
                    'station': station,
                    'passengers': passengers,
                    'temperature': weather['temperature'],
                    'weather_type': weather['weather_type'],
                    'is_weekend': is_weekend,
                    'events': [e['name'] for e in events]
                })
                
        return pd.DataFrame(data)
    
    def generate_dataset(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate passenger flow data for a date range"""
        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            daily_data = self.generate_daily_data(current_date)
            all_data.append(daily_data)
            current_date += timedelta(days=1)
            
        return pd.concat(all_data, ignore_index=True)

def main():
    """Example usage"""
    config = {
        'simulation_params': {
            'peak_hours': {
                'morning': {'start': '07:00', 'end': '09:00'},
                'evening': {'start': '16:00', 'end': '18:00'}
            }
        }
    }
    
    generator = PassengerFlowGenerator(config)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 7)
    
    df = generator.generate_dataset(start_date, end_date)
    print(f"Generated {len(df)} records")
    print("\nSample data:")
    print(df.head())
    
    # Basic statistics
    print("\nPassenger statistics by station:")
    print(df.groupby('station')['passengers'].describe())

if __name__ == "__main__":
    main()

#pip install optuna loguru --trusted-host pypi.org --trusted-host files.pythonhosted.org