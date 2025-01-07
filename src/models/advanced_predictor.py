# src/models/advanced_predictor.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

@dataclass
class ModelConfig:
    """Configuration for model architecture and training"""
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 10
    sequence_length: int = 24

class FeatureProcessor:
    """Process and transform features"""
    
    def __init__(self):
        self.num_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.numerical_columns = ['temperature']
        self.categorical_columns = ['weather_type']
        self.binary_columns = ['is_weekend']
        self.preprocessed_columns = []
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Process all features and return transformed dataframe with column names"""
        df = df.copy()
        
        # Process numerical features
        if self.numerical_columns:
            df[self.numerical_columns] = self.num_scaler.fit_transform(df[self.numerical_columns])
        self.preprocessed_columns.extend(self.numerical_columns)
        
        # Process categorical features
        for col in self.categorical_columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            self.preprocessed_columns.extend(dummies.columns)
        
        # Process binary features
        for col in self.binary_columns:
            df[col] = df[col].astype(int)
            self.preprocessed_columns.append(col)
        
        # Process events (if present)
        if 'events' in df.columns:
            df['has_event'] = df['events'].apply(lambda x: 1 if x and len(x) > 0 else 0)
            self.preprocessed_columns.append('has_event')
        
        # Scale target
        if 'passengers' in df.columns:
            df['passengers'] = self.target_scaler.fit_transform(df[['passengers']])
        
        return df, self.preprocessed_columns
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scalers"""
        df = df.copy()
        
        if self.numerical_columns:
            df[self.numerical_columns] = self.num_scaler.transform(df[self.numerical_columns])
        
        for col in self.categorical_columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        
        for col in self.binary_columns:
            df[col] = df[col].astype(int)
        
        if 'events' in df.columns:
            df['has_event'] = df['events'].apply(lambda x: 1 if x and len(x) > 0 else 0)
        
        return df

class TemporalDataset(Dataset):
    """Dataset for temporal data with sliding windows"""
    
    def __init__(
        self, 
        data: pd.DataFrame,
        sequence_length: int,
        feature_columns: List[str],
        target_column: str = 'passengers'
    ):
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        grouped = data.groupby('station')
        for _, station_data in grouped:
            # Sort and reset index for proper sequential data
            station_data = station_data.sort_values('datetime').reset_index(drop=True)
            
            # Ensure all required columns are present
            feature_data = station_data[feature_columns].astype(float)
            
            # Create sequences
            for i in range(len(station_data) - sequence_length):
                try:
                    sequence = feature_data.iloc[i:i + sequence_length].values
                    target = float(station_data.iloc[i + sequence_length][target_column])
                    
                    if sequence.shape[0] == sequence_length:
                        self.sequences.append(sequence)
                        self.targets.append(target)
                except Exception as e:
                    logger.warning(f"Error creating sequence: {str(e)}")
                    continue
        
        if not self.sequences:
            raise ValueError("No valid sequences could be created from the data")
        
        # Convert to numpy arrays with explicit dtype
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        
        # Convert to tensors
        self.sequences = torch.from_numpy(self.sequences).float()
        self.targets = torch.from_numpy(self.targets).float()
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]

class PassengerPredictor(nn.Module):
    """LSTM-based passenger prediction model"""
    
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1]).squeeze(-1)

class AdvancedPredictor:
    """Main predictor class with data processing and training logic"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_processor = FeatureProcessor()
        self.model = None
        self.feature_columns = None
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepare and scale data for training"""
        try:
            # Process features
            processed_df, self.feature_columns = self.feature_processor.fit_transform(df)
            
            # Create dataset
            dataset = TemporalDataset(
                processed_df,
                self.config.sequence_length,
                self.feature_columns,
                'passengers'
            )
            
            # Split data
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> pd.DataFrame:
        """Train the model"""
        input_dim = len(self.feature_columns)
        self.model = PassengerPredictor(input_dim, self.config).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(self.config.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(sequences)
                loss = criterion(predictions, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss = self.validate(val_loader, criterion)
            
            # Record history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pt'))
        return pd.DataFrame(training_history)
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(sequences)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)


    # In src/models/advanced_predictor.py, update the predict method:

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new data"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Process features
        processed_df = self.feature_processor.transform(df)
        predictions = np.zeros(len(df))
        
        # Process each station separately
        for station in df['station'].unique():
            station_mask = df['station'] == station
            station_data = processed_df[station_mask].reset_index(drop=True)
            
            if len(station_data) <= self.config.sequence_length:
                logger.warning(f"Insufficient data for station {station}")
                continue
            
            # Create sequences for prediction
            sequences = []
            indices = []
            
            for i in range(len(station_data) - self.config.sequence_length + 1):
                # Convert sequence to float32 immediately when creating
                sequence = station_data[self.feature_columns].iloc[i:i + self.config.sequence_length].values.astype(np.float32)
                sequences.append(sequence)
                indices.append(i + self.config.sequence_length - 1)
            
            if not sequences:
                continue
            
            # Stack sequences and convert to tensor
            sequences = np.stack(sequences, axis=0)
            sequences = torch.from_numpy(sequences).float().to(self.device)
            
            # Generate predictions in batches
            batch_size = self.config.batch_size
            station_predictions = []
            
            with torch.no_grad():
                for i in range(0, len(sequences), batch_size):
                    batch = sequences[i:i + batch_size]
                    batch_preds = self.model(batch).cpu().numpy()
                    station_predictions.extend(batch_preds)
            
            # Convert predictions to numpy array
            station_predictions = np.array(station_predictions)
            
            # Inverse transform predictions
            station_predictions = self.feature_processor.target_scaler.inverse_transform(
                station_predictions.reshape(-1, 1)
            ).flatten()
            
            # Assign predictions to the original dataframe indices
            original_indices = np.where(station_mask)[0][indices]
            predictions[original_indices] = station_predictions[:len(original_indices)]
        
        return predictions

def create_predictor() -> AdvancedPredictor:
    """Create a predictor with default configuration"""
    config = ModelConfig()
    return AdvancedPredictor(config)

if __name__ == "__main__":
    # Example usage
    from src.data.data_generator import PassengerFlowGenerator
    
    # Generate sample data
    config = {
        'simulation_params': {
            'peak_hours': {
                'morning': {'start': '07:00', 'end': '09:00'},
                'evening': {'start': '16:00', 'end': '18:00'}
            }
        }
    }
    
    generator = PassengerFlowGenerator(config)
    df = generator.generate_dataset(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 7)
    )
    
    # Train model
    predictor = create_predictor()
    train_loader, val_loader = predictor.prepare_data(df)
    history = predictor.train(train_loader, val_loader)
    
    # Make predictions
    predictions = predictor.predict(df)
    
    print("\nPrediction Statistics:")
    print(f"Mean prediction: {predictions.mean():.2f}")
    print(f"Min prediction: {predictions.min():.2f}")
    print(f"Max prediction: {predictions.max():.2f}")