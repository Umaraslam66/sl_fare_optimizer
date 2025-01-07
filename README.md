# SL Fare Optimization System

A machine learning system for dynamic fare optimization in Stockholm's public transport network.

## Features
- Dynamic fare prediction using Bidirectional LSTM
- Real-time demand forecasting
- Price optimization based on multiple factors
- Interactive visualizations
- Cross-validation and model evaluation

## Project Structure
```
sl_fare_optimizer/
├── data/               # Data storage
├── models/            # Trained models
├── logs/             # Application logs
├── results/          # Results and analysis
├── plots/            # Generated visualizations
├── src/              # Source code
│   ├── data/         # Data processing
│   ├── models/       # ML models
│   └── visualization/# Visualization tools
└── configs/          # Configuration files
```

## Installation
```bash
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
myenv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Usage
```bash
python main.py --days 30 --debug
```

## Configuration
Edit `configs/config.json` to modify:
- Base fare settings
- Model parameters
- Training configuration
- Visualization options

## Requirements
- Python 3.8+
- PyTorch
- Pandas
- NumPy
- Dash
- Plotly

## License
MIT