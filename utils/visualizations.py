import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_overview_charts(df):
    #Create overview charts for dashboard
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
    #Create time series chart showing events over time"""
    
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

def plot_feature_importance(model):
    #Plot feature importance from a trained RandomForest model.

    if not hasattr(model, "feature_importances_"):
        return None

    feat_cols = getattr(model, "feature_cols", [])
    importance = model.feature_importances_

    df = pd.DataFrame({
        "Feature": feat_cols,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance (RandomForest)",
        labels={"Importance": "Relative Importance", "Feature": "Feature"}
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    return fig


def create_impact_distribution(df):
    #Create impact level distribution chart
    
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
    #Create geographical analysis chart with proper regional data
    
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
    #Create event timeline visualization
    
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
    #Create cause analysis visualization
    
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
    #Create duration analysis charts
    
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
    #Create seasonal analysis visualization
    
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
    #Create comparison chart for multiple datasets
    
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

def create_seasonal_decomposition(df):
    #Create seasonal decomposition analysis using moving averages
    if df.empty or 'Date' not in df.columns:
        return go.Figure()
    
    # Prepare data - use event count by date
    df_temp = df.copy()
    df_temp['Date'] = pd.to_datetime(df_temp['Date'])
    daily_counts = df_temp.groupby(df_temp['Date'].dt.date).size()
    
    # Create a time series with regular frequency
    date_range = pd.date_range(start=daily_counts.index.min(), 
                              end=daily_counts.index.max(), 
                              freq='D')
    ts_data = daily_counts.reindex(date_range, fill_value=0)
    
    # Simple seasonal decomposition using moving averages
    window_size = 7  # weekly seasonality
    trend = ts_data.rolling(window=window_size, center=True).mean()
    seasonal = ts_data - trend
    residual = ts_data - (trend + seasonal)
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=['Original Time Series', 'Trend', 'Seasonal', 'Residual'],
        vertical_spacing=0.1
    )
    
    # Original data
    fig.add_trace(
        go.Scatter(x=ts_data.index, y=ts_data.values, name='Original'),
        row=1, col=1
    )
    
    # Trend component
    fig.add_trace(
        go.Scatter(x=trend.index, y=trend.values, name='Trend', line=dict(color='red')),
        row=2, col=1
    )
    
    # Seasonal component
    fig.add_trace(
        go.Scatter(x=seasonal.index, y=seasonal.values, name='Seasonal'),
        row=3, col=1
    )
    
    # Residual component
    fig.add_trace(
        go.Scatter(x=residual.index, y=residual.values, name='Residual'),
        row=4, col=1
    )
    
    fig.update_layout(height=600, title_text="Seasonal Decomposition Analysis")
    return fig

def detect_anomalies(df):
    #Detect anomalies in the dataset using IQR method
    if df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Check which columns to analyze
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        # If no numeric columns, use event counts by date
        if 'Date' in df.columns:
            df_temp = df.copy()
            df_temp['Date'] = pd.to_datetime(df_temp['Date'])
            daily_counts = df_temp.groupby(df_temp['Date'].dt.date).size().reset_index(name='Count')
            
            # Detect anomalies in daily counts
            Q1 = daily_counts['Count'].quantile(0.25)
            Q3 = daily_counts['Count'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify anomalies
            daily_counts['Anomaly'] = (daily_counts['Count'] < lower_bound) | (daily_counts['Count'] > upper_bound)
            
            # Create plot
            fig.add_trace(go.Scatter(
                x=daily_counts['Date'], 
                y=daily_counts['Count'],
                mode='markers',
                name='Daily Events',
                marker=dict(color='blue')
            ))
            
            # Highlight anomalies
            anomalies = daily_counts[daily_counts['Anomaly']]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies['Date'],
                    y=anomalies['Count'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=10, symbol='x')
                ))
            
            fig.update_layout(
                title="Anomaly Detection in Daily Event Counts",
                xaxis_title="Date",
                yaxis_title="Number of Events"
            )
        return fig
    
    # For each numeric column, detect anomalies
    for i, col in enumerate(numeric_cols[:3]):  # Limit to first 3 numeric columns
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify anomalies
        anomalies = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies.index if 'Date' not in df.columns else anomalies['Date'],
                y=anomalies[col],
                mode='markers',
                name=f'Anomalies in {col}',
                marker=dict(color='red', size=10, symbol='x')
            ))
    
    if len(fig.data) == 0:
        fig.add_annotation(text="No anomalies detected in the selected data",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
    
    fig.update_layout(title="Anomaly Detection Results")
    return fig

def create_risk_assessment(df):
    #Create risk assessment visualization
    if df.empty:
        return go.Figure()
    
    # Check if we have the required columns
    has_cause = 'Cause' in df.columns
    has_impact = 'Impact_Level' in df.columns
    
    if not has_cause or not has_impact:
        # Alternative visualization if missing required columns
        if 'Duration' in df.columns:
            # Show duration distribution by impact level if available
            impact_duration = df.groupby('Impact_Level' if has_impact else None)['Duration'].agg(['mean', 'count']).reset_index()
            fig = px.bar(
                impact_duration,
                x='Impact_Level' if has_impact else 'count',
                y='mean',
                title="Average Duration by Impact Level" if has_impact else "Duration Statistics"
            )
            return fig
        else:
            # Fallback: simple value counts of impact level
            impact_counts = df['Impact_Level'].value_counts() if has_impact else pd.Series([len(df)], index=['All Events'])
            fig = px.pie(
                values=impact_counts.values,
                names=impact_counts.index,
                title="Impact Level Distribution" if has_impact else "Event Count"
            )
            return fig
    
    # Create risk matrix if we have both Cause and Impact_Level
    risk_matrix = df.groupby(['Cause', 'Impact_Level']).size().reset_index(name='Count')
    
    fig = px.scatter(
        risk_matrix,
        x='Cause',
        y='Impact_Level',
        size='Count',
        color='Count',
        title="Risk Matrix: Cause vs Impact Level",
        color_continuous_scale='Reds'
    )
    
    return fig