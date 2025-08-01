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
    
    X, y, label_encoders, feature_cols = prepare_features(df)
    
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
    model.label_encoders = label_encoders
    model.feature_cols = feature_cols
    
    return model, accuracy, feature_importance, y_pred

def make_predictions(model, duration, region, cause, hour):
    """Make prediction for new data point"""
    
    if model is None:
        return "Model not trained"
    
    # Create feature vector
    features = {}
    
    # Numeric features
    features['Duration'] = duration
    features['Hour'] = hour
    features['Month'] = 6  # Default month
    features['DayOfWeek'] = 1  # Default day
    
    # Encode categorical features
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
