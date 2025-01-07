# src/visualization/dashboard.py

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path

class Dashboard:
    """Interactive dashboard for fare optimization visualization"""
    
    def __init__(self, config: dict):
        self.config = config
        self.app = dash.Dash(__name__)
        self.setup_layout()
        
    def setup_layout(self):
        """Create the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.H1("SL Fare Optimization Dashboard", 
                   className='text-center mb-4'),
            
            # Control Panel
            html.Div([
                html.H3("Controls"),
                html.Div([
                    html.Label("Select Station"),
                    dcc.Dropdown(id='station-select'),
                    
                    html.Label("Date Range"),
                    dcc.DatePickerRange(id='date-range'),
                    
                    html.Label("Update Frequency"),
                    dcc.Slider(
                        id='update-frequency',
                        min=1,
                        max=60,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(0, 61, 10)}
                    )
                ], className='p-4')
            ], className='mb-4'),
            
            # Graphs
            html.Div([
                # Passenger Flow
                html.Div([
                    html.H4("Passenger Flow"),
                    dcc.Graph(id='passenger-flow-graph')
                ], className='mb-4'),
                
                # Predictions
                html.Div([
                    html.H4("Predictions"),
                    dcc.Graph(id='predictions-graph')
                ], className='mb-4'),
                
                # Heatmap
                html.Div([
                    html.H4("Station Heatmap"),
                    dcc.Graph(id='station-heatmap')
                ], className='mb-4')
            ])
        ], className='container mx-auto p-4')
    
    def create_passenger_flow_figure(self, df: pd.DataFrame, station: str):
        """Create passenger flow visualization"""
        fig = go.Figure()
        
        station_data = df[df['station'] == station]
        
        fig.add_trace(go.Scatter(
            x=station_data['datetime'],
            y=station_data['passengers'],
            mode='lines',
            name='Actual Passengers'
        ))
        
        fig.update_layout(
            title=f'Passenger Flow at {station}',
            xaxis_title='Time',
            yaxis_title='Number of Passengers',
            hovermode='x unified'
        )
        
        return fig
    
    def create_predictions_figure(self, df: pd.DataFrame, station: str):
        """Create predictions visualization"""
        fig = go.Figure()
        
        station_data = df[df['station'] == station]
        
        fig.add_trace(go.Scatter(
            x=station_data['datetime'],
            y=station_data['passengers'],
            mode='lines',
            name='Actual'
        ))
        
        if 'predicted_passengers' in df.columns:
            fig.add_trace(go.Scatter(
                x=station_data['datetime'],
                y=station_data['predicted_passengers'],
                mode='lines',
                name='Predicted',
                line=dict(dash='dash')
            ))
        
        fig.update_layout(
            title=f'Predictions for {station}',
            xaxis_title='Time',
            yaxis_title='Number of Passengers',
            hovermode='x unified'
        )
        
        return fig
    
    def create_heatmap_figure(self, df: pd.DataFrame):
        """Create station heatmap visualization"""
        pivot_df = df.pivot_table(
            index='station',
            columns=df['datetime'].dt.hour,
            values='passengers',
            aggfunc='mean'
        )
        
        fig = px.imshow(
            pivot_df,
            labels=dict(x='Hour of Day', y='Station', color='Passengers'),
            aspect='auto'
        )
        
        fig.update_layout(
            title='Average Passenger Flow by Station and Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Station'
        )
        
        return fig
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            [Output('passenger-flow-graph', 'figure'),
             Output('predictions-graph', 'figure'),
             Output('station-heatmap', 'figure')],
            [Input('station-select', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_graphs(station, start_date, end_date):
            # Get data based on selection
            df = self.get_data(start_date, end_date)
            
            if station is None:
                station = df['station'].iloc[0]
            
            # Create figures
            flow_fig = self.create_passenger_flow_figure(df, station)
            pred_fig = self.create_predictions_figure(df, station)
            heat_fig = self.create_heatmap_figure(df)
            
            return flow_fig, pred_fig, heat_fig
    
    def get_data(self, start_date, end_date):
        """Get data for the selected date range"""
        # Implement data retrieval logic here
        pass
    
    def run(self, debug: bool = False, port: int = 8050):
        """Run the dashboard"""
        self.setup_callbacks()
        self.app.run_server(debug=debug, port=port)

def create_dashboard(config: dict) -> Dashboard:
    """Create a dashboard instance"""
    return Dashboard(config)