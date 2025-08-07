import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ---------------------------- Feature Preparation ----------------------------

def prepare_features(df):
    """Prepare the input features and labels for training a machine learning model."""
    if df.empty:
        return pd.DataFrame(), pd.Series()
    
    df_model = df.copy()

    # Convert Date and extract temporal features
    if 'Date' in df_model.columns:
        df_model['Date'] = pd.to_datetime(df_model['Date'])
        df_model['Year'] = df_model['Date'].dt.year
        df_model['Month'] = df_model['Date'].dt.month
        df_model['DayOfWeek'] = df_model['Date'].dt.dayofweek

    # Convert Time and extract Hour
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

    # Select numeric + encoded features
    feature_cols = ['Duration', 'Hour', 'Month', 'DayOfWeek']
    for col in categorical_cols:
        if col in df_model.columns:
            feature_cols.append(f'{col}_encoded')

    # Ensure only columns present in the data are used
    feature_cols = [col for col in feature_cols if col in df_model.columns]

    X = df_model[feature_cols]
    y = df_model['Impact_Level'] if 'Impact_Level' in df_model.columns else pd.Series()

    X = X.fillna(X.mean() if not X.empty else 0)

    return X, y, label_encoders, feature_cols

# ---------------------------- Model Training ----------------------------

def train_random_forest(df, n_estimators=100, max_depth=10, test_size=0.2):
    """Train a Random Forest model on the dataset."""
    result = prepare_features(df)
    if len(result) == 2:
        X, y = result
        label_encoders, feature_cols = {}, []
    else:
        X, y, label_encoders, feature_cols = result

    if X.empty or y.empty:
        return None, 0, [], []

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Define and train the model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Feature importance
    feature_importance = [{
        'feature': col,
        'importance': model.feature_importances_[i]
    } for i, col in enumerate(feature_cols)]
    feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)

    # Save encoders and feature columns
    setattr(model, 'label_encoders', label_encoders)
    setattr(model, 'feature_cols', feature_cols)

    return model, accuracy, feature_importance, y_pred

# ---------------------------- Prediction ----------------------------

def make_predictions(model, duration, region, cause, hour):
    """Make prediction using the trained model for given inputs."""
    if model is None:
        return "Model not trained"

    features = {
        'Duration': duration,
        'Hour': hour,
        'Month': 6,
        'DayOfWeek': 1
    }

    # Encode Region and Cause
    if hasattr(model, 'label_encoders'):
        if 'Region' in model.label_encoders:
            try:
                features['Region_encoded'] = model.label_encoders['Region'].transform([region])[0]
            except ValueError:
                features['Region_encoded'] = 0
        if 'Cause' in model.label_encoders:
            try:
                features['Cause_encoded'] = model.label_encoders['Cause'].transform([cause])[0]
            except ValueError:
                features['Cause_encoded'] = 0

    feature_df = pd.DataFrame([features])

    for col in model.feature_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0

    feature_df = feature_df[model.feature_cols]

    prediction = model.predict(feature_df)[0]
    return prediction

# ---------------------------- 72-Hour ML Prediction ----------------------------

def generate_72_hour_predictions(models_data, all_data):
    """Generate real ML-based predictions for the next 72 hours using trained models."""
    from datetime import datetime, timedelta
    import plotly.graph_objects as go

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

    for dataset_key, info in dataset_info.items():
        if dataset_key in all_data and not all_data[dataset_key].empty:
            df = all_data[dataset_key]

            # Re-train or reuse the trained model
            model_info = train_random_forest(df)
            model, _, _, _ = model_info
            if model is None:
                continue

            predictions = []
            for future_time in prediction_times:
                hour = future_time.hour
                month = future_time.month
                dayofweek = future_time.weekday()

                # Create sample input based on feature averages
                sample_row = {
                    'Duration': df['Duration'].mean() if 'Duration' in df.columns else 1.0,
                    'Hour': hour,
                    'Month': month,
                    'DayOfWeek': dayofweek
                }

                # Encode region and cause
                if hasattr(model, 'label_encoders'):
                    sample_row['Region_encoded'] = 0
                    sample_row['Cause_encoded'] = 0

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

# ---------------------------- Model Insights ----------------------------

def get_model_insights(model, X, y):
    """Generate metrics and insights about the model performance."""
    if model is None:
        return {}

    feature_importance = dict(zip(model.feature_cols, model.feature_importances_))
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    class_distribution = pd.Series(y).value_counts().to_dict()

    insights = {
        'accuracy': accuracy,
        'feature_importance': feature_importance,
        'class_distribution': class_distribution,
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth
    }

    return insights

# ---------------------------- Cross Validation ----------------------------

def cross_validate_model(df, cv_folds=5):
    """Cross-validation for performance evaluation."""
    from sklearn.model_selection import cross_val_score

    X, y, _, _ = prepare_features(df)
    if X.empty or y.empty:
        return []

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')

    return scores

# ---------------------------- Feature Ranking ----------------------------

def feature_selection(df, top_k=10):
    """Select the top k most important features using Random Forest."""
    X, y, label_encoders, feature_cols = prepare_features(df)
    if X.empty or y.empty:
        return []

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    importance_scores = list(zip(feature_cols, model.feature_importances_))
    importance_scores = sorted(importance_scores, key=lambda x: x[1], reverse=True)

    return importance_scores[:top_k]
