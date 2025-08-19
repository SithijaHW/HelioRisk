import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Mapping between labels and numeric ticks for plotting
IMPACT_TO_NUM = {'Low': 1, 'Medium': 2, 'High': 3}
NUM_TO_IMPACT = {v: k for k, v in IMPACT_TO_NUM.items()}


# ---------------------------- Dataset builder ----------------------------

def build_combined_dataset(all_data):
    """
    Combine data from provided dict of DataFrames into a single dataset suitable for training.
    Returns: X, y, encoders, feature_cols
    """
    frames = []
    keys_to_check = ['power_grid', 'gps_disruptions', 'solar_flare', 'solar_wind', 'satellite']

    for key in keys_to_check:
        if key in all_data and not all_data[key].empty:
            df = all_data[key].copy()

            # Need Impact_Level to train
            if 'Impact_Level' not in df.columns:
                continue

            # Normalize datetimes
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce')

            # Temporal features with safe defaults
            df['Hour'] = df['Time'].dt.hour.fillna(12).astype(int) if 'Time' in df.columns else 12
            df['Month'] = df['Date'].dt.month.fillna(datetime.now().month).astype(int) if 'Date' in df.columns else datetime.now().month
            df['DayOfWeek'] = df['Date'].dt.weekday.fillna(1).astype(int) if 'Date' in df.columns else 1

            # Duration
            if 'Duration' in df.columns:
                df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
                if df['Duration'].isna().all():
                    df['Duration'] = 1.0
                else:
                    df['Duration'] = df['Duration'].fillna(df['Duration'].median())
            else:
                df['Duration'] = 1.0

            # Region & Cause
            df['Region'] = df.get('Region', pd.Series(['Unknown'] * len(df))).fillna('Unknown').astype(str)
            df['Cause'] = df.get('Cause', pd.Series(['Unknown'] * len(df))).fillna('Unknown').astype(str)

            df['source'] = key

            # Standardize Impact_Level and keep only known classes
            df['Impact_Level'] = df['Impact_Level'].astype(str).str.strip().str.title()
            df = df[df['Impact_Level'].isin(['Low', 'Medium', 'High'])]

            frames.append(df[['Duration', 'Hour', 'Month', 'DayOfWeek', 'Region', 'Cause', 'source', 'Impact_Level']])

    if not frames:
        return pd.DataFrame(), pd.Series(dtype=object), {}, []

    combined = pd.concat(frames, ignore_index=True)

    # Encode categorical columns
    encoders = {}
    for col in ['Region', 'Cause', 'source']:
        le = LabelEncoder()
        combined[f'{col}_encoded'] = le.fit_transform(combined[col].astype(str))
        encoders[col] = le

    feature_cols = ['Duration', 'Hour', 'Month', 'DayOfWeek', 'Region_encoded', 'Cause_encoded', 'source_encoded']

    X = combined[feature_cols].copy()
    y = combined['Impact_Level'].copy()

    return X, y, encoders, feature_cols


# ---------------------------- Training ----------------------------

def train_combined_model(all_data, n_estimators=100, max_depth=10, test_size=0.2, save_path=None):
    """
    Train a RandomForest on combined datasets.
    Returns: model, accuracy, encoders, feature_cols
    """
    X, y, encoders, feature_cols = build_combined_dataset(all_data)
    if X.empty or y.empty:
        return None, 0.0, {}, []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Attach metadata
    model.feature_cols = feature_cols
    model.label_encoders = encoders

    if save_path:
        try:
            payload = {'model': model, 'encoders': encoders, 'feature_cols': feature_cols}
            joblib.dump(payload, save_path)
        except Exception:
            pass  # saving is optional; don't crash if disk issues

    return model, accuracy, encoders, feature_cols


def train_random_forest(df_or_all_data, n_estimators=100, max_depth=10, test_size=0.2, save_path=None):
    """
    Backwards-compatible wrapper: accepts either a single dataframe (legacy) or the whole dict of dataframes.
    """
    if isinstance(df_or_all_data, dict):
        return train_combined_model(df_or_all_data, n_estimators=n_estimators, max_depth=max_depth, test_size=test_size, save_path=save_path)
    # If single dataframe provided, wrap into dict as 'power_grid'
    return train_combined_model({'power_grid': df_or_all_data}, n_estimators=n_estimators, max_depth=max_depth, test_size=test_size, save_path=save_path)


# ---------------------------- Utilities for safe row creation ----------------------------

def _safe_row_dict(feature_cols, values_dict):
    """
    Build a 1-row DataFrame that has exactly feature_cols columns.
    Missing values are filled with 0 (or sensible defaults if provided in values_dict).
    """
    if not feature_cols:
        # default feature columns (safe fallback)
        feature_cols = ['Duration', 'Hour', 'Month', 'DayOfWeek', 'Region_encoded', 'Cause_encoded', 'source_encoded']
    row = {col: values_dict.get(col, 0) for col in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)


# ---------------------------- Single-sample predictions ----------------------------

def make_predictions(model, duration, region, cause, hour, source='power_grid'):
    """
    Make a single prediction using a trained model (must have label_encoders & feature_cols attached).
    Returns predicted label string ('Low'|'Medium'|'High') or error message.
    """
    if model is None:
        return "Model not trained"

    # Raw values with sensible defaults
    sample = {
        'Duration': float(duration) if duration is not None else 1.0,
        'Hour': int(hour) if hour is not None else 12,
        'Month': datetime.now().month,
        'DayOfWeek': datetime.now().weekday(),
        'Region': str(region) if region is not None else 'Unknown',
        'Cause': str(cause) if cause is not None else 'Unknown',
        'source': str(source) if source is not None else 'power_grid'
    }

    encs = getattr(model, 'label_encoders', {})
    # Add encoded fields using encoders; unseen -> fallback to 0
    for col in ['Region', 'Cause', 'source']:
        encoded_key = f'{col}_encoded'
        le = encs.get(col)
        if le:
            try:
                sample[encoded_key] = int(le.transform([sample[col]])[0])
            except Exception:
                sample[encoded_key] = 0
        else:
            sample[encoded_key] = 0

    feat_cols = getattr(model, 'feature_cols', None)
    if not feat_cols:
        feat_cols = ['Duration', 'Hour', 'Month', 'DayOfWeek', 'Region_encoded', 'Cause_encoded', 'source_encoded']

    feature_df = _safe_row_dict(feat_cols, sample)

    try:
        pred = model.predict(feature_df)[0]
    except Exception:
        # In unlikely case prediction fails, fallback to Low
        pred = 'Low'

    return pred


# ---------------------------- 72-hour predictions ----------------------------

def generate_72_hour_predictions(model_or_data, all_data=None):
    """
    Optimized 72-hour prediction generator.
    - Builds batched DataFrames (one per system) and predicts all 72 rows at once.
    - Avoids repeated retraining if model already exists.
    """
    # Determine inputs
    if isinstance(model_or_data, dict):
        all_data = model_or_data
        model = None
    else:
        model = model_or_data

    # If no model is provided, train only once
    if model is None:
        if not isinstance(all_data, dict) or not any(k in all_data and not all_data[k].empty for k in ['power_grid','gps_disruptions','solar_flare','solar_wind','satellite']):
            fig = go.Figure()
            fig.update_layout(title="72-Hour Space Weather Impact Predictions (No training data available)")
            return fig
        model, _, _, _ = train_combined_model(all_data)
        if model is None:
            fig = go.Figure()
            fig.update_layout(title="72-Hour Space Weather Impact Predictions (Failed to train model)")
            return fig

    # Prediction times (72 hours)
    now = datetime.now()
    prediction_times = [now + timedelta(hours=i) for i in range(72)]

    # Systems to predict for
    dataset_keys = []
    if isinstance(all_data, dict):
        dataset_keys = [k for k in ['power_grid','gps_disruptions','solar_flare','solar_wind','satellite']
                        if k in all_data and not all_data[k].empty]

    if not dataset_keys:
        encs = getattr(model, 'label_encoders', {})
        src_le = encs.get('source')
        dataset_keys = list(src_le.classes_) if src_le is not None else ['combined']

    feat_cols = getattr(model, 'feature_cols', ['Duration','Hour','Month','DayOfWeek','Region_encoded','Cause_encoded','source_encoded'])
    encs = getattr(model, 'label_encoders', {})

    fig = go.Figure()

    for key in dataset_keys:
        # Determine representative values from dataset
        if isinstance(all_data, dict) and key in all_data and not all_data[key].empty:
            df = all_data[key]
            dur = float(df['Duration'].mean()) if 'Duration' in df.columns else 1.0
            region = df['Region'].mode()[0] if 'Region' in df.columns and not df['Region'].mode().empty else 'Unknown'
            cause = df['Cause'].mode()[0] if 'Cause' in df.columns and not df['Cause'].mode().empty else 'Unknown'
        else:
            dur, region, cause = 1.0, 'Unknown', 'Unknown'

        # Build all 72 rows at once
        samples = []
        for t in prediction_times:
            vals = {
                'Duration': dur,
                'Hour': t.hour,
                'Month': t.month,
                'DayOfWeek': t.weekday(),
                'Region': region,
                'Cause': cause,
                'source': key
            }
            # Encoded fields
            for col in ['Region','Cause','source']:
                le = encs.get(col)
                enc_key = f"{col}_encoded"
                if le:
                    try:
                        vals[enc_key] = int(le.transform([vals[col]])[0])
                    except Exception:
                        vals[enc_key] = 0
                else:
                    vals[enc_key] = 0
            samples.append(vals)

        batch_df = pd.DataFrame([{c: v.get(c,0) for c in feat_cols} for v in samples], columns=feat_cols)

        # Single batched predict
        try:
            pred_labels = model.predict(batch_df)
        except Exception:
            pred_labels = ['Low'] * len(batch_df)

        pred_nums = [IMPACT_TO_NUM.get(str(lbl).title(),1) for lbl in pred_labels]

        # Add one line trace
        fig.add_trace(go.Scatter(
            x=prediction_times,
            y=pred_nums,
            mode='lines+markers',
            name=(key if key != 'combined' else 'Combined'),
            line=dict(width=2),
            marker=dict(size=4)
        ))

    fig.update_layout(
        title="72-Hour Space Weather Impact Predictions by System (ML-based, Optimized)",
        xaxis_title="Time",
        yaxis_title="Predicted Impact Level",
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['Low','Medium','High'],
            range=[0.5, 3.5]
        ),
        height=480,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ---------------------------- Utilities ----------------------------

def load_saved_model(path):
    """Load a saved model payload from joblib (returns model or None)"""
    try:
        payload = joblib.load(path)
        model = payload.get('model')
        if model is None:
            return None
        # Restore metadata
        model.feature_cols = payload.get('feature_cols', getattr(model, 'feature_cols', None))
        model.label_encoders = payload.get('encoders', getattr(model, 'label_encoders', {}))
        return model
    except Exception:
        return None
