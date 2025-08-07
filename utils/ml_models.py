import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def prepare_features(df):
    """Prepare features for machine learning model"""
    if df.empty:
        return pd.DataFrame(), pd.Series()
    
    # Create a copy to avoid modifying original data
    df_model = df.copy()
    
    # Extract features from datetime
    if 'Date' in df_model.columns:
        df_model['Date'] = pd.to_datetime(df_model['Date'])
        df_model['Year'] = df_model['Date'].dt.year
        df_model['Month'] = df_model['Date'].dt.month
        df_model['DayOfWeek'] = df_model['Date'].dt.dayofweek
    
    if 'Time' in df_model.columns:
        df_model['Time'] = pd.to_datetime(df_model['Time'], format='%H:%M', errors='coerce')
        df_model['Hour'] = df_model['Time'].dt.hour
    
    # Encode categorical variables
    categorical_cols = ['Region', 'Cause']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[f'{col}_encoded'] = le.fit_transform(df_model[col].astype(str))
            label_encoders[col] = le
    
    # Select features for model
    feature_cols = ['Duration', 'Hour', 'Month', 'DayOfWeek']
    
    # Add encoded categorical features
    for col in categorical_cols:
        if col in df_model.columns:
            feature_cols.append(f'{col}_encoded')
    
    # Filter existing columns
    feature_cols = [col for col in feature_cols if col in df_model.columns]
    
    X = df_model[feature_cols]
    y = df_model['Impact_Level'] if 'Impact_Level' in df_model.columns else pd.Series()
    
    # Handle missing values
    X = X.fillna(X.mean() if not X.empty else 0)
    
    return X, y, label_encoders, feature_cols

def train_random_forest(df, n_estimators=100, max_depth=10, test_size=0.2):
    """Train Random Forest model for impact level prediction"""
    
    result = prepare_features(df)
    if len(result) == 2:
        X, y = result
        label_encoders, feature_cols = {}, []
    else:
        X, y, label_encoders, feature_cols = result
    
    if X.empty or y.empty:
        return None, 0, [], []
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = []
    for i, col in enumerate(feature_cols):
        feature_importance.append({
            'feature': col,
            'importance': model.feature_importances_[i]
        })
    
    feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
    
    # Store encoders in model for later use
    setattr(model, 'label_encoders', label_encoders)
    setattr(model, 'feature_cols', feature_cols)
    
    return model, accuracy, feature_importance, y_pred

def make_predictions(model, duration, region, cause, hour):
    """Make prediction for new data point"""
    
    if model is None:
        return "Model not trained"
    # Create feature vector
    # Encode categorical features
    features = {
        'Duration': duration,
        'Hour': hour,
        'Month': 6,
        'DayOfWeek': 1
    }
    if hasattr(model, 'label_encoders'):
        if 'Region' in model.label_encoders:
            try:
                features['Region_encoded'] = model.label_encoders['Region'].transform([region])[0]
            except ValueError:
                # Handle unseen category
                features['Region_encoded'] = 0
        
        if 'Cause' in model.label_encoders:
            try:
                features['Cause_encoded'] = model.label_encoders['Cause'].transform([cause])[0]
            except ValueError:
                # Handle unseen category
                features['Cause_encoded'] = 0
    
    # Create DataFrame with correct column order
    feature_df = pd.DataFrame([features])
    
    # Ensure all required columns are present
    for col in model.feature_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0
    
    # Reorder columns to match training data
    feature_df = feature_df[model.feature_cols]
    
    # Make prediction
    prediction = model.predict(feature_df)[0]
    prediction_proba = model.predict_proba(feature_df)[0]
    
    return prediction

def generate_72_hour_predictions(models_data, all_data):
    """Generate predictions for the next 72 hours for each dataset"""
    from datetime import datetime, timedelta
    import plotly.graph_objects as go
    
    # Available datasets for prediction
    dataset_info = {
        'power_grid': {'name': 'Power Grid', 'color': '#E74C3C'},
        'gps_disruptions': {'name': 'GPS Systems', 'color': '#3498DB'},
        'solar_flare': {'name': 'Solar Flares', 'color': '#F39C12'},
        'solar_wind': {'name': 'Solar Wind', 'color': '#27AE60'},
        'satellite': {'name': 'Satellites', 'color': '#9B59B6'}
    }
    
    current_time = datetime.now()
    prediction_times = [current_time + timedelta(hours=i) for i in range(72)]
    
    fig = go.Figure()
    
    # Generate predictions for each available dataset
    for dataset_key, info in dataset_info.items():
        if dataset_key in all_data and not all_data[dataset_key].empty:
            # Create more realistic predictions with temporal patterns
            df = all_data[dataset_key]
            model_info = train_random_forest(df)
            model, _, _, _ = model_info
            if model is None:
                continue
            predictions = []
            
                # Create more balanced prediction logic
            for future_time in prediction_times:
                hour = future_time.hour
            
                month = future_time.month
                dayofweek = future_time.weekday()
                sample_row = {
                    'Duration': df['Duration'].mean() if 'Duration' in df.columns else 1.0,
                    'Hour': hour,
                    'Month': month,
                    'DayOfWeek': dayofweek
                }
                if hasattr(model, 'label_encoders'):
                    region_encoder = model.label_encoders.get('Region')
                    cause_encoder = model.label_encoders.get('Cause')
                    sample_row['Region_encoded'] = 0 if region_encoder is None else 0
                    sample_row['Cause_encoded'] = 0 if cause_encoder is None else 0
                row_df = pd.DataFrame([sample_row])
                for col in model.feature_cols:
                    if col not in row_df.columns:
                        row_df[col] = 0
                row_df = row_df[model.feature_cols]
                impact = model.predict(row_df)[0]
                predictions.append(impact)
            fig.add_trace(go.Scatter(
                x=prediction_times,
                y=predictions,
                mode='lines+markers',
                name=info['name'],
                line=dict(color=info['color'], width=2),
                marker=dict(size=4)
            ))
    
    fig.update_layout(
        title="72-Hour Space Weather Impact Predictions by System",
        xaxis_title="Time",
        yaxis_title="Predicted Impact Level",
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['Low', 'Medium', 'High'],
            range=[0.5, 3.5]
        ),
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def get_model_insights(model, X, y):
    """Get detailed model insights and performance metrics"""
    
    if model is None:
        return {}
    
    # Feature importance analysis
    feature_importance = dict(zip(model.feature_cols, model.feature_importances_))
    
    # Model performance
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    # Class distribution
    class_distribution = pd.Series(y).value_counts().to_dict()
    
    insights = {
        'accuracy': accuracy,
        'feature_importance': feature_importance,
        'class_distribution': class_distribution,
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth
    }
    
    return insights

def cross_validate_model(df, cv_folds=5):
    """Perform cross-validation on the model"""
    from sklearn.model_selection import cross_val_score
    
    X, y, _, _ = prepare_features(df)
    
    if X.empty or y.empty:
        return []
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
    
    return scores

def feature_selection(df, top_k=10):
    """Select top k most important features"""
    
    X, y, label_encoders, feature_cols = prepare_features(df)
    
    if X.empty or y.empty:
        return []
    
    # Train a simple model to get feature importance
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Get feature importance
    importance_scores = list(zip(feature_cols, model.feature_importances_))
    importance_scores = sorted(importance_scores, key=lambda x: x[1], reverse=True)
    
    return importance_scores[:top_k]
