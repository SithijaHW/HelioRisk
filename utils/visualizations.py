import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_overview_charts(df):
    """Create overview charts for dashboard"""
    charts = {}
    
    if df.empty:
        return charts
    
    # Impact level distribution
    if 'Impact_Level' in df.columns:
        impact_counts = df['Impact_Level'].value_counts()
        charts['impact_pie'] = px.pie(
            values=impact_counts.values,
            names=impact_counts.index,
            title="Impact Level Distribution",
            color_discrete_map={
                'Low': '#27AE60',
                'Medium': '#F39C12',
                'High': '#E74C3C'
            }
        )
    
    return charts

def create_time_series_chart(df):
    """Create time series chart showing events over time"""
    
    if df.empty or 'Date' not in df.columns:
        return go.Figure()
    
    # Prepare data
    df_temp = df.copy()
    df_temp['Date'] = pd.to_datetime(df_temp['Date'])
    
    # Group by date and impact level
    daily_events = df_temp.groupby([df_temp['Date'].dt.date, 'Impact_Level']).size().reset_index(name='Count')
    daily_events['Date'] = pd.to_datetime(daily_events['Date'])
    
    # Create figure
    fig = px.line(
        daily_events,
        x='Date',
        y='Count',
        color='Impact_Level',
        title="Events Over Time by Impact Level",
        color_discrete_map={
            'Low': '#27AE60',
            'Medium': '#F39C12',
            'High': '#E74C3C'
        }
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Events",
        hovermode='x unified'
    )
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap for numeric columns"""
    
    if df.empty:
        return go.Figure()
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return go.Figure()
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        title="Correlation Heatmap",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    
    fig.update_layout(
        title="Feature Correlation Analysis",
        width=600,
        height=500
    )
    
    return fig

def create_impact_distribution(df):
    """Create impact level distribution chart"""
    
    if df.empty or 'Impact_Level' not in df.columns:
        return go.Figure()
    
    impact_counts = df['Impact_Level'].value_counts()
    
    fig = px.bar(
        x=impact_counts.index,
        y=impact_counts.values,
        title="Impact Level Distribution",
        color=impact_counts.index,
        color_discrete_map={
            'Low': '#27AE60',
            'Medium': '#F39C12',
            'High': '#E74C3C'
        }
    )
    
    fig.update_layout(
        xaxis_title="Impact Level",
        yaxis_title="Number of Events",
        showlegend=False
    )
    
    return fig

def create_geographical_analysis(df, data_type='power_grid'):
    """Create geographical analysis chart with proper regional data"""
    
    if df.empty:
        return go.Figure()
    
    # Handle different regional columns based on data type
    region_col = None
    if 'Region' in df.columns:
        region_col = 'Region'
    elif 'Location' in df.columns:
        region_col = 'Location'
    
    if region_col is None:
        return go.Figure().add_annotation(
            text="No regional data available for this dataset",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Regional distribution
    regional_data = df.groupby([region_col, 'Impact_Level']).size().reset_index(name='Count')
    
    fig = px.bar(
        regional_data,
        x=region_col,
        y='Count',
        color='Impact_Level',
        title=f"Events by {region_col} and Impact Level",
        color_discrete_map={
            'Low': '#27AE60',
            'Medium': '#F39C12',
            'High': '#E74C3C'
        }
    )
    
    fig.update_layout(
        xaxis_title=region_col,
        yaxis_title="Number of Events",
        barmode='stack'
    )
    
    return fig

def create_event_timeline(df):
    """Create event timeline visualization"""
    
    if df.empty or 'Date' not in df.columns:
        return go.Figure()
    
    # Get recent events (last 50)
    df_temp = df.copy()
    df_temp['Date'] = pd.to_datetime(df_temp['Date'])
    df_recent = df_temp.nlargest(50, 'Date')
    
    # Create timeline
    fig = px.scatter(
        df_recent,
        x='Date',
        y='Duration' if 'Duration' in df_recent.columns else 'Impact_Level',
        color='Impact_Level',
        size='Duration' if 'Duration' in df_recent.columns else None,
        hover_data=['Region', 'Cause'] if 'Region' in df_recent.columns else None,
        title="Recent Event Timeline",
        color_discrete_map={
            'Low': '#27AE60',
            'Medium': '#F39C12',
            'High': '#E74C3C'
        }
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Duration (hours)" if 'Duration' in df_recent.columns else "Impact Level",
        height=400
    )
    
    return fig

def create_cause_analysis(df):
    """Create cause analysis visualization"""
    
    if df.empty or 'Cause' not in df.columns:
        return go.Figure()
    
    cause_impact = df.groupby(['Cause', 'Impact_Level']).size().reset_index(name='Count')
    
    fig = px.sunburst(
        cause_impact,
        path=['Cause', 'Impact_Level'],
        values='Count',
        title="Event Causes and Impact Levels"
    )
    
    return fig

def create_duration_analysis(df):
    """Create duration analysis charts"""
    
    if df.empty or 'Duration' not in df.columns:
        return go.Figure()
    
    # Duration distribution by impact level
    fig = px.box(
        df,
        x='Impact_Level',
        y='Duration',
        title="Duration Distribution by Impact Level",
        color='Impact_Level',
        color_discrete_map={
            'Low': '#27AE60',
            'Medium': '#F39C12',
            'High': '#E74C3C'
        }
    )
    
    fig.update_layout(
        xaxis_title="Impact Level",
        yaxis_title="Duration (hours)"
    )
    
    return fig

def create_seasonal_analysis(df):
    """Create seasonal analysis visualization"""
    
    if df.empty or 'Date' not in df.columns:
        return go.Figure()
    
    df_temp = df.copy()
    df_temp['Date'] = pd.to_datetime(df_temp['Date'])
    df_temp['Month'] = df_temp['Date'].dt.month_name()
    df_temp['Season'] = df_temp['Date'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    seasonal_data = df_temp.groupby(['Season', 'Impact_Level']).size().reset_index(name='Count')
    
    fig = px.bar(
        seasonal_data,
        x='Season',
        y='Count',
        color='Impact_Level',
        title="Seasonal Event Distribution",
        color_discrete_map={
            'Low': '#27AE60',
            'Medium': '#F39C12',
            'High': '#E74C3C'
        }
    )
    
    return fig

def create_multi_dataset_comparison(datasets):
    """Create comparison chart for multiple datasets"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Power Grid Events', 'Satellite Anomalies', 'Solar Flares', 'Solar Wind Events'],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Define colors for each dataset
    colors = ['#4A90E2', '#E74C3C', '#F39C12', '#27AE60']
    
    for i, (name, df) in enumerate(datasets.items()):
        if not df.empty and 'Date' in df.columns:
            df_temp = df.copy()
            df_temp['Date'] = pd.to_datetime(df_temp['Date'])
            daily_counts = df_temp.groupby(df_temp['Date'].dt.date).size()
            
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=list(daily_counts.index),
                    y=list(daily_counts.values),
                    mode='lines+markers',
                    name=name.replace('_', ' ').title(),
                    line=dict(color=colors[i])
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title="Multi-Dataset Event Comparison",
        height=600,
        showlegend=True
    )
    
    return fig
