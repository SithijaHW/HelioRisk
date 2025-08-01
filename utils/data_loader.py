import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_all_data():
    """Load all CSV files and return as dictionary"""
    data = {}
    
    # File paths
    files = {
        'power_grid': 'attached_assets/power_grid_failures_1754061801584.csv',
        'satellite': 'attached_assets/satellite_anomalies_1754061801585.csv',
        'solar_flare': 'attached_assets/solar_flare_data_1754061801586.csv',
        'solar_wind': 'attached_assets/solar_wind_data_1754061801586.csv',
        'gps_disruptions': 'attached_assets/gps_disruptions_1754061801582.csv',
        'cme_data': 'attached_assets/cme_data_1754061801582.csv',
        'kp_index': 'attached_assets/kp_index_data_1754061801583.csv'
    }
    
    for key, file_path in files.items():
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                data[key] = preprocess_data(df, key)
            else:
                print(f"Warning: {file_path} not found")
                data[key] = pd.DataFrame()
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            data[key] = pd.DataFrame()
    
    return data

def preprocess_data(df, data_type):
    """Preprocess data based on type"""
    if df.empty:
        return df
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert date columns
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Handle missing values
    if data_type == 'power_grid':
        # Ensure proper data types for power grid data
        if 'Duration' in df.columns:
            df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
        
        # Clean impact levels
        if 'Impact_Level' in df.columns:
            df['Impact_Level'] = df['Impact_Level'].str.strip().str.title()
            df['Impact_Level'] = df['Impact_Level'].replace({'Low': 'Low', 'Medium': 'Medium', 'High': 'High'})
    
    elif data_type == 'satellite':
        # Process satellite anomaly data
        if 'Related_Event' in df.columns:
            df['Related_Event'] = df['Related_Event'].str.strip()
    
    elif data_type == 'solar_flare':
        # Process solar flare data
        if 'Peak_Flux' in df.columns:
            df['Peak_Flux'] = pd.to_numeric(df['Peak_Flux'], errors='coerce')
        if 'Duration' in df.columns:
            df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    
    elif data_type == 'solar_wind':
        # Process solar wind data
        numeric_cols = ['Speed', 'Density', 'Bz']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    elif data_type == 'gps_disruptions':
        # Process GPS disruption data
        if 'Duration' in df.columns:
            df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
        if 'Region' in df.columns:
            df['Region'] = df['Region'].str.strip()
    
    elif data_type == 'cme_data':
        # Process CME data
        numeric_cols = ['Speed', 'Width']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'Direction' in df.columns:
            df['Direction'] = df['Direction'].str.strip()
    
    elif data_type == 'kp_index':
        # Process Kp index data
        if 'Kp_Index' in df.columns:
            df['Kp_Index'] = pd.to_numeric(df['Kp_Index'], errors='coerce')
        if 'Duration' in df.columns:
            df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    
    # Remove rows with critical missing values
    df = df.dropna(subset=['Date'] if 'Date' in df.columns else [])
    
    return df

def get_summary_statistics(data):
    """Generate summary statistics for all datasets"""
    summary = {}
    
    for key, df in data.items():
        if not df.empty:
            summary[key] = {
                'total_events': len(df),
                'date_range': {
                    'start': df['Date'].min() if 'Date' in df.columns else None,
                    'end': df['Date'].max() if 'Date' in df.columns else None
                }
            }
            
            if 'Impact_Level' in df.columns:
                summary[key]['impact_distribution'] = df['Impact_Level'].value_counts().to_dict()
            
            if 'Duration' in df.columns:
                summary[key]['avg_duration'] = df['Duration'].mean()
                summary[key]['max_duration'] = df['Duration'].max()
    
    return summary

def filter_data_by_date(data, start_date, end_date):
    """Filter all datasets by date range"""
    filtered_data = {}
    
    for key, df in data.items():
        if not df.empty and 'Date' in df.columns:
            mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
            filtered_data[key] = df.loc[mask]
        else:
            filtered_data[key] = df
    
    return filtered_data

def get_regional_statistics(df):
    """Get statistics by region"""
    if df.empty or 'Region' not in df.columns:
        return {}
    
    regional_stats = {}
    for region in df['Region'].unique():
        region_data = df[df['Region'] == region]
        regional_stats[region] = {
            'total_events': len(region_data),
            'high_impact': len(region_data[region_data.get('Impact_Level', '') == 'High']),
            'avg_duration': region_data['Duration'].mean() if 'Duration' in region_data.columns else 0
        }
    
    return regional_stats
