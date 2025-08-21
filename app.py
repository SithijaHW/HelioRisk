from streamlit_autorefresh import st_autorefresh
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import utility modules
from utils.data_loader import load_all_data, preprocess_data
from utils.ml_models import train_random_forest, make_predictions, generate_72_hour_predictions
from utils.ml_models import (train_random_forest, make_predictions, generate_72_hour_predictions, load_saved_model)
from utils.visualizations import (
    create_overview_charts, create_time_series_chart, 
    plot_feature_importance, create_impact_distribution,
    create_geographical_analysis, create_event_timeline
)
from utils.pdf_generator import generate_pdf_report
from utils.educational_content import get_educational_content

# Page configuration
st.set_page_config(
    page_title="Helio Risk - Space Weather Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for light theme and professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4A90E2 0%, #74B3F0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .metric-card {
        background: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4A90E2;
        margin: 0.5rem 0;
    }
    
    .tab-content {
        padding: 1rem 0;
    }
    
    .alert-high {
        background: #FFE6E6;
        border-left: 4px solid #E74C3C;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .alert-medium {
        background: #FFF3CD;
        border-left: 4px solid #F39C12;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .alert-low {
        background: #E8F5E8;
        border-left: 4px solid #27AE60;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #E3F2FD;
        border: 1px solid #BBDEFB;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_current_alerts(data):
    """Check datasets and return threshold alerts using simulated streaming."""
    alerts = []

    # Increment index for streaming effect
    st.session_state["alert_index"] += 1
    idx = st.session_state["alert_index"]

    # --- Power Grid ---
    if 'power_grid' in data and not data['power_grid'].empty:
        df = data['power_grid'].sort_values("Date")
        row = df.iloc[idx % len(df)]   # cycle through rows
        if row.get("Impact_Level") == "High":
            alerts.append({
                "level": "High",
                "infrastructure": "Power Grid",
                "region": row.get("Region", "Unknown"),
                "time": row.get("Date", "")
            })

    # --- Kp Index ---
    if 'kp_index' in data and not data['kp_index'].empty:
        df = data['kp_index'].sort_values("Date")
        row = df.iloc[idx % len(df)]
        kp_val = row.get("Kp_Index", 0)
        if kp_val > 6:
            alerts.append({
                "level": "High",
                "infrastructure": "Geomagnetic Field",
                "region": "Global",
                "time": row.get("Date", "")
            })
        elif kp_val > 4:
            alerts.append({
                "level": "Medium",
                "infrastructure": "Geomagnetic Field",
                "region": "Global",
                "time": row.get("Date", "")
            })

    # --- Solar Wind ---
    if 'solar_wind' in data and not data['solar_wind'].empty:
        df = data['solar_wind'].sort_values("Date")
        row = df.iloc[idx % len(df)]
        speed = row.get("Speed", 0)
        bz = row.get("Bz", 0)
        if speed > 700 or bz < -10:
            alerts.append({
                "level": "High",
                "infrastructure": "Solar Wind",
                "region": "Global",
                "time": row.get("Date", "")
            })
        elif speed > 500:
            alerts.append({
                "level": "Medium",
                "infrastructure": "Solar Wind",
                "region": "Global",
                "time": row.get("Date", "")
            })

    # --- Solar Flare ---
    if 'solar_flare' in data and not data['solar_flare'].empty:
        df = data['solar_flare'].sort_values("Date")
        row = df.iloc[idx % len(df)]
        flux = row.get("Peak_Flux", 0)
        if flux > 1.0:
            alerts.append({
                "level": "High",
                "infrastructure": "Solar Flare",
                "region": "Global",
                "time": row.get("Date", "")
            })
        elif flux > 0.1:
            alerts.append({
                "level": "Medium",
                "infrastructure": "Solar Flare",
                "region": "Global",
                "time": row.get("Date", "")
            })

    return alerts


        # Initialize streaming index for simulated alerts
if "alert_index" not in st.session_state:
    st.session_state["alert_index"] = 0

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌌 Helio Risk</h1>
        <h3>Comprehensive Space Weather Monitoring & Risk Assessment Dashboard</h3>
        <p>Advanced Analytics • Machine Learning Predictions • Real-time Monitoring</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    @st.cache_data
    def load_data():
        return load_all_data()
    
    if 'model' not in st.session_state:
        st.session_state['model'] = load_saved_model('model_payload.pkl')

    
    try:
        data = load_data()

            # --- Notification Bar (Hybrid) ---
        threshold_alerts = get_current_alerts(data)
        ml_alerts = get_ml_alerts(st.session_state.get("model"), data)
        all_alerts = sorted(threshold_alerts + ml_alerts, key=lambda x: x["time"], reverse=True)



        if all_alerts:
            latest = all_alerts[0]
            source_tag = "🔮 ML" if latest.get("source") == "ML Prediction" else "⚡ Observed"
            st.markdown(f"""
            <div style="background:#FFE6E6;padding:0.7rem;border-radius:8px;margin-bottom:1rem;">
                ⚠️ <b>{latest['level']} Impact</b> | {latest['infrastructure']} | Region: {latest['region']} | {latest['time']} | {source_tag}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#E8F5E8;padding:0.7rem;border-radius:8px;margin-bottom:1rem;">
                ✅ No critical alerts at this time.
            </div>
            """, unsafe_allow_html=True)


        
        # Sidebar for global filters
        with st.sidebar:
            st.markdown("### 🎛️ Global Filters")
            
            # Date range filter
            if not data['power_grid'].empty:
                min_date = pd.to_datetime(data['power_grid']['Date']).min().date()
                max_date = pd.to_datetime(data['power_grid']['Date']).max().date()
                
                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            
            # Impact level filter
            impact_levels = st.multiselect(
                "Impact Levels",
                options=['Low', 'Medium', 'High'],
                default=['Low', 'Medium', 'High']
            )
            
            # Infrastructure filter
            infrastructure_types = st.multiselect(
                "Infrastructure Types",
                options=['Power Grid', 'Satellite', 'Solar Flare', 'Solar Wind'],
                default=['Power Grid', 'Satellite', 'Solar Flare', 'Solar Wind']
            )
            
            st.markdown("---")
            st.markdown("### 📊 Quick Stats")
            
            # Quick statistics
            total_events = 0
            for df in data.values():
                if not df.empty:
                    total_events += len(df)
            
            st.metric("Total Events", f"{total_events:,}")
            
            if not data['power_grid'].empty:
                high_impact = len(data['power_grid'][data['power_grid']['Impact_Level'] == 'High'])
                st.metric("High Impact Events", high_impact)
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🏠 Overview", "🤖 ML Predictions", "📚 Educational", 
            "📄 Download Center", "📈 Data Analysis", "🎯 Interactive Viz", "🚨 Alert System"
        ])
        
        with tab1:
            overview_dashboard(data, date_range if 'date_range' in locals() else None, impact_levels, infrastructure_types)
        
        with tab2:
            ml_predictions_tab(data)
        
        with tab3:
            educational_tab()
        
        with tab4:
            download_center_tab(data)
        
        with tab5:
            data_analysis_tab(data)
        
        with tab6:
            interactive_viz_tab(data)
        
        with tab7:
            alert_system_tab(data)
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure all CSV files are available in the attached_assets folder.")

def overview_dashboard(data, date_range, impact_levels, infrastructure_types):
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("## 🌍 Space Weather Overview Dashboard")
    
    # Key metrics row - First row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'power_grid' in data and not data['power_grid'].empty:
            power_events = len(data['power_grid'])
            st.markdown(f"""
            <div class="metric-card">
                <h4>⚡ Power Grid Events</h4>
                <h2>{power_events:,}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'satellite' in data and not data['satellite'].empty:
            sat_events = len(data['satellite'])
            st.markdown(f"""
            <div class="metric-card">
                <h4>🛰️ Satellite Anomalies</h4>
                <h2>{sat_events:,}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'solar_flare' in data and not data['solar_flare'].empty:
            flare_events = len(data['solar_flare'])
            st.markdown(f"""
            <div class="metric-card">
                <h4>☀️ Solar Flares</h4>
                <h2>{flare_events:,}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'solar_wind' in data and not data['solar_wind'].empty:
            wind_events = len(data['solar_wind'])
            st.markdown(f"""
            <div class="metric-card">
                <h4>🌊 Solar Wind Events</h4>
                <h2>{wind_events:,}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Second row for new datasets
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'gps_disruptions' in data and not data['gps_disruptions'].empty:
            gps_events = len(data['gps_disruptions'])
            st.markdown(f"""
            <div class="metric-card">
                <h4>📡 GPS Disruptions</h4>
                <h2>{gps_events:,}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'cme_data' in data and not data['cme_data'].empty:
            cme_events = len(data['cme_data'])
            st.markdown(f"""
            <div class="metric-card">
                <h4>💥 CME Events</h4>
                <h2>{cme_events:,}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'kp_index' in data and not data['kp_index'].empty:
            kp_events = len(data['kp_index'])
            st.markdown(f"""
            <div class="metric-card">
                <h4>🌍 Geomagnetic Events</h4>
                <h2>{kp_events:,}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Charts
    st.markdown("### 📊 Event Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Impact level distribution
        if not data['power_grid'].empty:
            fig = create_impact_distribution(data['power_grid'])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Events over time
        if not data['power_grid'].empty:
            fig = create_time_series_chart(data['power_grid'])
            st.plotly_chart(fig, use_container_width=True)
    
    # Regional analysis
    st.markdown("### 🌍 Regional Impact Analysis")
    
    # Create tabs for different data types with regional data
    regional_tab1, regional_tab2 = st.columns(2)
    
    with regional_tab1:
        st.markdown("#### Power Grid Failures by Region")
        if 'power_grid' in data and not data['power_grid'].empty:
            fig = create_geographical_analysis(data['power_grid'], 'power_grid')
            st.plotly_chart(fig, use_container_width=True)
    
    with regional_tab2:
        st.markdown("#### GPS Disruptions by Region")
        if 'gps_disruptions' in data and not data['gps_disruptions'].empty:
            fig = create_geographical_analysis(data['gps_disruptions'], 'gps_disruptions')
            st.plotly_chart(fig, use_container_width=True)
    
    # Event timeline
    st.markdown("### ⏰ Recent Event Timeline")
    if not data['power_grid'].empty:
        fig = create_event_timeline(data['power_grid'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def ml_predictions_tab(data):
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("## 🤖 Machine Learning Predictions")
    st_autorefresh(interval=60000, key="72_hour_refresh")
    
    st.markdown("""
    <div class="info-box">
        <h4>🧠 Predictive Analytics Engine</h4>
        <p>Our Random Forest model analyzes space weather patterns to predict impact levels 
        based on various parameters including solar activity, geomagnetic conditions, and historical patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎯 Model Configuration")
        
        # Model parameters
        n_estimators = st.slider("Number of Trees", 50, 500, 100)
        max_depth = st.slider("Max Depth", 3, 20, 10)
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training Random Forest model on all datasets..."):
                model, accuracy, _, _ = train_random_forest(
                    data, n_estimators=n_estimators, max_depth=max_depth, test_size=test_size, save_path='model_payload.pkl'
                )
                if model:
                    st.session_state['model'] = model
                    st.session_state['accuracy'] = accuracy
                    st.success(f"✅ Combined model trained and saved! Accuracy: {accuracy:.2%}")
                else:
                    st.error("❌ Model training failed — check if your datasets have Impact_Level labels.")
    
    with col2:
        st.markdown("### 📈 Model Performance")
        
        if 'accuracy' in st.session_state:
            # Accuracy gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = st.session_state['accuracy'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Model Accuracy (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#4A90E2"},
                    'steps': [
                        {'range': [0, 50], 'color': "#FFE6E6"},
                        {'range': [50, 80], 'color': "#FFF3CD"},
                        {'range': [80, 100], 'color': "#E8F5E8"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            if 'feature_importance' in st.session_state:
                st.markdown("### 🎯 Feature Importance")
                importance_df = pd.DataFrame(st.session_state['feature_importance'])
                fig = px.bar(
                    importance_df, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title="Feature Importance in Predictions",
                    color='importance',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Prediction interface

    st.markdown("### 🔮 Make Predictions")

    date = st.date_input("Select a Date for Prediction")

    if st.button("🎯 Predict Impact", type="primary"):
        if 'model' in st.session_state and st.session_state['model'] is not None:
            results = make_predictions(st.session_state['model'], date, data)

            for res in results:
                impact_level = res["Impact_Level"]
                infra = res["Infrastructure"]
                region = res["Region"]

                if impact_level == 'High':
                    st.markdown(f"""
                    <div class="alert-high">
                        <h4>⚠️ {infra} Impact: HIGH RISK</h4>
                        <p>Region at risk: <b>{region}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                elif impact_level == 'Medium':
                    st.markdown(f"""
                    <div class="alert-medium">
                        <h4>⚡ {infra} Impact: MEDIUM RISK</h4>
                        <p>Region at risk: <b>{region}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                elif impact_level == 'Low':
                    st.markdown(f"""
                    <div class="alert-low">
                        <h4>✅ {infra} Impact: LOW RISK</h4>
                        <p>Region at risk: <b>{region}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.info(f"{infra}: Prediction unavailable")
        else:
            st.warning("⚠️ Please train the model first!")

    
    # 72-Hour Continuous Predictions
    st.markdown("### 🔮 72-Hour Continuous Impact Predictions")
    
    if 'model' in st.session_state:
        st.markdown("""
        <div class="info-box">
            <h4>📈 Continuous Forecasting</h4>
            <p>This chart shows predicted impact levels for the next 72 hours based on historical patterns and current space weather conditions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate and display 72-hour predictions
        with st.spinner("Generating 72-hour predictions..."):
            prediction_chart = generate_72_hour_predictions(st.session_state['model'], all_data=data)
            st.plotly_chart(prediction_chart, use_container_width=True)
            
        # Auto-refresh option
        auto_refresh = st.checkbox("🔄 Auto-refresh predictions every 1 minute")
        
        if auto_refresh:
            st.markdown("""
            <div class="alert-low">
                <p>🔄 Predictions will automatically update every 1 minute to reflect the latest space weather conditions.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # JavaScript for auto-refresh every minute
    else:
        st.info("Train the model first to see 72-hour continuous predictions.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def educational_tab():
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("## 📚 Space Weather Education Center")
    
    content = get_educational_content()
    
    # Educational sections
    for section in content:
        with st.expander(f"🔍 {section['title']}", expanded=False):
            st.markdown(section['content'])
            
            if 'image_url' in section:
                st.image(section['image_url'], caption=section.get('image_caption', ''))
            
            if 'video_url' in section:
                st.video(section['video_url'])
    
    # Interactive learning quiz
    st.markdown("### 🧠 Test Your Knowledge")
    
    quiz_questions = [
        {
            "question": "What is a Coronal Mass Ejection (CME)?",
            "options": [
                "A solar flare that releases electromagnetic radiation",
                "A large expulsion of plasma and magnetic field from the Sun's corona",
                "A temporary reduction in solar wind speed",
                "A type of geomagnetic storm"
            ],
            "correct": 1
        },
        {
            "question": "Which space weather event is most likely to affect satellite operations?",
            "options": [
                "Solar wind variations",
                "Geomagnetic storms",
                "Solar flares",
                "All of the above"
            ],
            "correct": 3
        }
    ]
    
    for i, quiz in enumerate(quiz_questions):
        st.markdown(f"**Question {i+1}:** {quiz['question']}")
        answer = st.radio(f"Select answer for question {i+1}:", quiz['options'], key=f"quiz_{i}")
        
        if st.button(f"Check Answer {i+1}", key=f"check_{i}"):
            if quiz['options'].index(answer) == quiz['correct']:
                st.success("✅ Correct! Well done!")
            else:
                st.error(f"❌ Incorrect. The correct answer is: {quiz['options'][quiz['correct']]}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def download_center_tab(data):
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("## 📄 Download Center")
    
    st.markdown("""
    <div class="info-box">
        <h4>📋 Custom Report Generation</h4>
        <p>Generate comprehensive PDF reports with customizable filters and parameters. 
        Perfect for stakeholders, compliance, and detailed analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Report Configuration")
        
        # Report parameters
        report_title = st.text_input("Report Title", "Helio Risk - Space Weather Analysis Report")
        
        # Date range for report
        if not data['power_grid'].empty:
            min_date = pd.to_datetime(data['power_grid']['Date']).min().date()
            max_date = pd.to_datetime(data['power_grid']['Date']).max().date()
            
            report_date_range = st.date_input(
                "Report Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        # Filters for report
        report_events = st.multiselect(
            "Include Event Types",
            options=['Power Grid Failures', 'Satellite Anomalies', 'Solar Flares', 'Solar Wind Events'],
            default=['Power Grid Failures', 'Satellite Anomalies']
        )
        
        report_impact = st.multiselect(
            "Impact Levels",
            options=['Low', 'Medium', 'High'],
            default=['Medium', 'High']
        )
        
        include_charts = st.checkbox("Include Charts and Visualizations", value=True)
        include_predictions = st.checkbox("Include ML Predictions", value=True)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
    
    with col2:
        st.markdown("### 📊 Report Preview")
        
        # Show preview statistics
        if not data['power_grid'].empty:
            filtered_data = data['power_grid'][
                data['power_grid']['Impact_Level'].isin(report_impact)
            ]
            
            st.metric("Events in Report", len(filtered_data))
            st.metric("High Risk Events", len(filtered_data[filtered_data['Impact_Level'] == 'High']))
            
            # Preview chart
            if len(filtered_data) > 0:
                fig = px.pie(
                    filtered_data, 
                    names='Impact_Level', 
                    title="Impact Level Distribution (Preview)",
                    color_discrete_map={
                        'Low': '#27AE60',
                        'Medium': '#F39C12', 
                        'High': '#E74C3C'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Generate report button
    st.markdown("### 📥 Generate Report")
    
    if st.button("🚀 Generate PDF Report", type="primary"):
        with st.spinner("Generating comprehensive PDF report..."):
            try:
                # Filter data based on selections
                filtered_data = {}
                for key, df in data.items():
                    if not df.empty and 'Impact_Level' in df.columns:
                        filtered_data[key] = df[df['Impact_Level'].isin(report_impact)]
                    else:
                        filtered_data[key] = df
                
                # Generate PDF
                pdf_buffer = generate_pdf_report(
                    filtered_data, 
                    report_title, 
                    report_date_range if 'report_date_range' in locals() else None,
                    include_charts,
                    include_predictions,
                    include_recommendations
                )
                
                # Download button
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name=f"helio_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
                
                st.success("✅ Report generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    # Quick download options
    st.markdown("### ⚡ Quick Downloads")
    
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("📊 Download Data (CSV)"):
            if not data['power_grid'].empty:
                csv = data['power_grid'].to_csv(index=False)
                st.download_button(
                    label="Download Power Grid Data",
                    data=csv,
                    file_name="power_grid_data.csv",
                    mime="text/csv"
                )
    
    with quick_col2:
        if st.button("📈 Download Charts (PNG)"):
            st.info("Chart export functionality - implementation depends on specific chart selection")
    
    with quick_col3:
        if st.button("📋 Download Summary (TXT)"):
            if not data['power_grid'].empty:
                summary = f"""
                Helio Risk - Summary Report
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                Total Events: {len(data['power_grid'])}
                High Impact: {len(data['power_grid'][data['power_grid']['Impact_Level'] == 'High'])}
                Medium Impact: {len(data['power_grid'][data['power_grid']['Impact_Level'] == 'Medium'])}
                Low Impact: {len(data['power_grid'][data['power_grid']['Impact_Level'] == 'Low'])}
                """
                
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name="helio_risk_summary.txt",
                    mime="text/plain"
                )
    
    st.markdown('</div>', unsafe_allow_html=True)

def data_analysis_tab(data):
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("## 📈 Advanced Data Analysis")
    
    # Correlation analysis
    st.markdown("### 🔗 Which factors matter most for prediction?")

    if 'model' in st.session_state and st.session_state['model'] is not None:
        fig = plot_feature_importance(st.session_state['model'])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train a model to see feature importance.")
    else:
        st.warning("Please train the model first to see feature importance.")

    
    # Time series analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Temporal Patterns")
        
        if not data['power_grid'].empty:
            # Monthly distribution
            df_temp = data['power_grid'].copy()
            df_temp['Date'] = pd.to_datetime(df_temp['Date'])
            df_temp['Month'] = df_temp['Date'].dt.month_name()
            
            monthly_counts = df_temp.groupby('Month').size().reset_index(name='Count')
            
            fig = px.bar(
                monthly_counts,
                x='Month',
                y='Count',
                title="Events by Month",
                color='Count',
                color_continuous_scale='Blues'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### ⏰ Hourly Distribution")
        
        if not data['power_grid'].empty:
            df_temp = data['power_grid'].copy()
            df_temp['Time'] = pd.to_datetime(df_temp['Time'], format='%H:%M')
            df_temp['Hour'] = df_temp['Time'].dt.hour
            
            hourly_counts = df_temp.groupby('Hour').size().reset_index(name='Count')
            
            fig = px.line(
                hourly_counts,
                x='Hour',
                y='Count',
                title="Events by Hour of Day",
                markers=True
            )
            fig.update_traces(line_color='#4A90E2', marker_color='#4A90E2')
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistical insights
    st.markdown("### 📊 Statistical Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        if not data['power_grid'].empty:
            avg_duration = data['power_grid']['Duration'].mean()
            st.metric("Average Duration", f"{avg_duration:.1f} hours")
    
    with insight_col2:
        if not data['power_grid'].empty:
            most_common_cause = data['power_grid']['Cause'].mode()[0]
            st.metric("Most Common Cause", most_common_cause)
    
    with insight_col3:
        if not data['power_grid'].empty:
            most_affected_region = data['power_grid']['Region'].mode()[0]
            st.metric("Most Affected Region", most_affected_region)
    
    # Advanced analytics
    st.markdown("### 🧮 Advanced Analytics")
    
    analysis_option = st.selectbox(
        "Select Analysis Type",
        ["Trend Analysis", "Seasonal Decomposition", "Anomaly Detection", "Risk Assessment"]
    )
    
    if analysis_option == "Trend Analysis" and not data['power_grid'].empty:
        st.markdown("#### 📈 Long-term Trends")
        
        df_temp = data['power_grid'].copy()
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])
        df_temp['Year'] = df_temp['Date'].dt.year
        
        yearly_trends = df_temp.groupby(['Year', 'Impact_Level']).size().reset_index(name='Count')
        
        fig = px.line(
            yearly_trends,
            x='Year',
            y='Count',
            color='Impact_Level',
            title="Event Trends by Impact Level Over Years",
            color_discrete_map={
                'Low': '#27AE60',
                'Medium': '#F39C12',
                'High': '#E74C3C'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_option == "Risk Assessment":
        st.markdown("#### ⚠️ Risk Assessment Matrix")
        
        if not data['power_grid'].empty:
            risk_matrix = data['power_grid'].groupby(['Cause', 'Impact_Level']).size().reset_index(name='Count')
            
            fig = px.scatter(
                risk_matrix,
                x='Cause',
                y='Impact_Level',
                size='Count',
                color='Count',
                title="Risk Matrix: Cause vs Impact Level",
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def interactive_viz_tab(data):
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("## 🎯 Interactive Visualizations")
    
    # Dataset selector
    st.markdown("### 📂 Select Dataset")
    available_datasets = [k for k, v in data.items() if not v.empty]
    dataset_choice = st.selectbox("Choose dataset", available_datasets + ["Combined"])
    
    # Prepare data
    if dataset_choice == "Combined":
        # Combine datasets with common columns
        combined_frames = []
        for k, df in data.items():
            if not df.empty:
                temp = df.copy()
                temp['Dataset'] = k
                combined_frames.append(temp)
        if combined_frames:
            df = pd.concat(combined_frames, ignore_index=True)
        else:
            st.warning("No datasets available to combine.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
    else:
        df = data[dataset_choice].copy()
    
    if df.empty:
        st.info("No data available for visualization.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Ensure Time -> Hour exists
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce')
        df['Hour'] = df['Time'].dt.hour
    
    # Encode categorical columns
    categorical_cols = ['Region', 'Cause', 'Impact_Level', 'Dataset']
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            df[f'{col}_encoded'] = pd.Categorical(df[col])
            encoders[col] = dict(enumerate(df[f'{col}_encoded'].cat.categories))
            df[f'{col}_encoded'] = df[f'{col}_encoded'].cat.codes
    
    # User axis selection
    st.markdown("### 🎛️ Dynamic Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        viz_type = st.selectbox(
            "Visualization Type",
            ["Scatter Plot", "3D Analysis", "Bubble Chart", "Sunburst", "Treemap"]
        )
    
    with filter_col2:
        x_axis = st.selectbox(
            "X-Axis",
            [c for c in ['Duration', 'Region', 'Cause', 'Impact_Level', 'Hour', 'Dataset'] if c in df.columns]
        )
    
    with filter_col3:
        y_axis = st.selectbox(
            "Y-Axis",
            [c for c in ['Duration', 'Impact_Level', 'Cause', 'Region', 'Hour', 'Dataset'] if c in df.columns]
        )
    
    def get_axis(axis):
        return f"{axis}_encoded" if axis in encoders else axis
    
    x_col = get_axis(x_axis)
    y_col = get_axis(y_axis)
    
    # ---- Scatter Plot ----
    if viz_type == "Scatter Plot":
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color='Impact_Level' if 'Impact_Level' in df.columns else None,
            hover_data=[c for c in ['Region', 'Cause', 'Dataset'] if c in df.columns],
            title=f"{dataset_choice}: {x_axis} vs {y_axis}"
        )
        if x_axis in encoders:
            fig.update_xaxes(tickmode='array',
                             tickvals=list(encoders[x_axis].keys()),
                             ticktext=list(encoders[x_axis].values()))
        if y_axis in encoders:
            fig.update_yaxes(tickmode='array',
                             tickvals=list(encoders[y_axis].keys()),
                             ticktext=list(encoders[y_axis].values()))
    
    # ---- 3D Analysis ----
    elif viz_type == "3D Analysis":
        if 'Hour' not in df.columns:
            st.warning("Hour column not available for 3D Analysis.")
            return
        fig = px.scatter_3d(
            df,
            x='Hour',
            y='Duration' if 'Duration' in df.columns else x_col,
            z='Impact_Level_encoded' if 'Impact_Level' in df.columns else y_col,
            color='Impact_Level' if 'Impact_Level' in df.columns else None,
            hover_data=[c for c in ['Region', 'Cause', 'Dataset'] if c in df.columns],
            title=f"{dataset_choice}: 3D Analysis"
        )
    
    # ---- Sunburst ----
    elif viz_type == "Sunburst":
        path_cols = [c for c in ['Region', 'Cause', 'Impact_Level', 'Dataset'] if c in df.columns]
        if len(path_cols) >= 2:
            fig = px.sunburst(df, path=path_cols, title=f"{dataset_choice}: Hierarchical Breakdown")
        else:
            st.warning("Not enough categorical columns for Sunburst.")
            return
    
    # ---- Treemap ----
    elif viz_type == "Treemap":
        group_cols = [c for c in ['Region', 'Impact_Level', 'Dataset'] if c in df.columns]
        if group_cols:
            impact_counts = df.groupby(group_cols).size().reset_index(name='Count')
            fig = px.treemap(
                impact_counts,
                path=group_cols,
                values='Count',
                title=f"{dataset_choice}: Events Breakdown Treemap",
                color='Count',
                color_continuous_scale='Blues'
            )
        else:
            st.warning("Not enough columns for Treemap.")
            return
    
    # ---- Bubble Chart ----
    else:
        if 'Hour' not in df.columns or 'Duration' not in df.columns:
            st.warning("Hour/Duration column not available for Bubble Chart.")
            return
        bubble_data = df.groupby([c for c in ['Region', 'Impact_Level', 'Dataset'] if c in df.columns]).agg({
            'Duration': 'mean',
            'Hour': 'mean'
        }).reset_index()
        bubble_data['Count'] = df.groupby([c for c in ['Region', 'Impact_Level', 'Dataset'] if c in df.columns]).size().values
        
        fig = px.scatter(
            bubble_data,
            x='Hour',
            y='Duration',
            size='Count',
            color='Impact_Level' if 'Impact_Level' in df.columns else None,
            hover_name='Region' if 'Region' in bubble_data.columns else None,
            title=f"{dataset_choice}: Bubble Chart (Avg Duration vs Hour)"
        )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Multi-dataset comparison
    st.markdown("### 📊 Multi-Dataset Comparison")
    
    if not data['satellite'].empty and not data['power_grid'].empty:
        # Compare satellite anomalies with power grid events
        sat_df = data['satellite'].copy()
        power_df = data['power_grid'].copy()
        
        # Time series comparison
        sat_df['Date'] = pd.to_datetime(sat_df['Date'])
        power_df['Date'] = pd.to_datetime(power_df['Date'])
        
        sat_daily = sat_df.groupby(sat_df['Date'].dt.date).size()
        power_daily = power_df.groupby(power_df['Date'].dt.date).size()
        
        # Align dates
        all_dates = sorted(set(sat_daily.index) | set(power_daily.index))
        sat_aligned = [sat_daily.get(date, 0) for date in all_dates]
        power_aligned = [power_daily.get(date, 0) for date in all_dates]
        
        comparison_df = pd.DataFrame({
            'Date': all_dates,
            'Satellite_Events': sat_aligned,
            'Power_Grid_Events': power_aligned
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=comparison_df['Date'],
            y=comparison_df['Satellite_Events'],
            mode='lines+markers',
            name='Satellite Anomalies',
            line=dict(color='#E74C3C')
        ))
        fig.add_trace(go.Scatter(
            x=comparison_df['Date'],
            y=comparison_df['Power_Grid_Events'],
            mode='lines+markers',
            name='Power Grid Events',
            line=dict(color='#4A90E2')
        ))
        
        fig.update_layout(
            title="Daily Event Comparison: Satellite vs Power Grid",
            xaxis_title="Date",
            yaxis_title="Number of Events",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def get_ml_alerts(model, data):
    """Generate ML-based alerts using simulated streaming for all datasets."""
    alerts = []
    if model is None:
        return alerts

    idx = st.session_state["alert_index"]

    for dataset_key, df in data.items():
        if df.empty:
            continue

        df = df.sort_values("Date")
        row = df.iloc[idx % len(df)]

        try:
            duration = row.get("Duration", 1)
            region = row.get("Region", "Unknown")
            cause = row.get("Cause", "Unknown")
            hour = (pd.to_datetime(row.get("Time", "12:00"), errors='coerce').hour
                    if "Time" in row else 12)

            pred = make_predictions(model, duration, region, cause, hour)

            alerts.append({
                "level": pred,
                "infrastructure": dataset_key.replace("_", " ").title(),
                "region": region,
                "time": row.get("Date", ""),
                "source": "ML Prediction"
            })
        except Exception as e:
            print(f"ML alert failed for {dataset_key}: {e}")
            continue

    return alerts

def alert_system_tab(data):
    st_autorefresh(interval=60000, key="alert_refresh")  # refresh every 60s

    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("## 🚨 Real-time Alert System")

    threshold_alerts = get_current_alerts(data)
    ml_alerts = get_ml_alerts(st.session_state.get("models", {}), data)
    all_alerts = sorted(threshold_alerts + ml_alerts, key=lambda x: x["time"], reverse=True)


    # Current Alerts Overview
    st.markdown("### 🌐 Current Alerts Overview")
    if not all_alerts:
        st.success("✅ No active alerts at this time.")
    else:
        for alert in all_alerts[:10]:  # Show top 10 alerts
            level_class = "alert-high" if alert["level"] == "High" else "alert-medium" if alert["level"] == "Medium" else "alert-low"
            source_tag = "🔮 ML Prediction" if alert.get("source") == "ML Prediction" else "⚡ Observed"
            st.markdown(f"""
            <div class="{level_class}">
                <h4>{alert['level']} Impact - {alert['infrastructure']}</h4>
                <p><b>Region:</b> {alert['region']} | <b>Time:</b> {alert['time']} | {source_tag}</p>
            </div>
            """, unsafe_allow_html=True)

    # Detailed Dataset Checks
    st.markdown("### 📊 Detailed Status by Dataset")

    if 'power_grid' in data and not data['power_grid'].empty:
        st.markdown("#### ⚡ Power Grid Events")
        latest = data['power_grid'].sort_values("Date").iloc[-1]
        st.write(latest)

    if 'kp_index' in data and not data['kp_index'].empty:
        st.markdown("#### 🌍 Geomagnetic Activity (Kp Index)")
        latest = data['kp_index'].sort_values("Date").iloc[-1]
        st.write(latest)

    if 'solar_wind' in data and not data['solar_wind'].empty:
        st.markdown("#### 🌊 Solar Wind Conditions")
        latest = data['solar_wind'].sort_values("Date").iloc[-1]
        st.write(latest)

    if 'solar_flare' in data and not data['solar_flare'].empty:
        st.markdown("#### ☀️ Solar Flare Data")
        latest = data['solar_flare'].sort_values("Date").iloc[-1]
        st.write(latest)

    # Recent High-Impact Events (Threshold only)
    st.markdown("### ⏰ Recent High-Impact Events (Observed)")
    if 'power_grid' in data and not data['power_grid'].empty:
        df = data['power_grid'].copy()
        df = df[df['Impact_Level'] == 'High'].sort_values("Date", ascending=False)
        limit = st.slider("Number of events to display", 5, 50, 10)
        if not df.empty:
            st.dataframe(df.head(limit))
        else:
            st.info("No recent high-impact power grid events.")

    st.markdown('</div>', unsafe_allow_html=True)

    
    # Alert configuration
    st.markdown("### ⚙️ Alert Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.markdown("#### 📧 Notification Settings")
        
        email_alerts = st.checkbox("Email Alerts", value=True)
        sms_alerts = st.checkbox("SMS Alerts", value=False)
        
        if email_alerts:
            email_address = st.text_input("Email Address", "admin@heliorisk.com")
        
        alert_threshold = st.selectbox(
            "Alert Threshold",
            ["High Impact Only", "Medium & High Impact", "All Events"]
        )
    
    with config_col2:
        st.markdown("#### 🎯 Alert Criteria")
        
        duration_alert = st.number_input("Alert if duration exceeds (hours)", min_value=1, value=8)
        
        regions_alert = st.multiselect(
            "Alert for regions",
            ["USA", "Europe", "China", "India"],
            default=["USA", "Europe"]
        )
        
        causes_alert = st.multiselect(
            "Alert for causes",
            ["Overload", "Hardware Fault", "Geomagnetic Disturbance"],
            default=["Geomagnetic Disturbance"]
        )
    
    # Monitoring dashboard
    st.markdown("### 📊 Live Monitoring Dashboard")
    
    # Real-time style metrics (simulated)
    monitor_col1, monitor_col2, monitor_col3, monitor_col4 = st.columns(4)
    
    with monitor_col1:
        st.metric("Active Alerts", "3", delta="1")
    
    with monitor_col2:
        st.metric("System Health", "98%", delta="2%")
    
    with monitor_col3:
        st.metric("Data Latency", "1.2s", delta="-0.3s")
    
    with monitor_col4:
        st.metric("Prediction Accuracy", "94%", delta="1%")
    
    # Alert history chart
    if not data['power_grid'].empty:
        st.markdown("### 📈 Alert Frequency Trends")
        
        df = data['power_grid'].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Week'] = df['Date'].dt.isocalendar().week
        
        weekly_alerts = df[df['Impact_Level'].isin(['Medium', 'High'])].groupby('Week').size()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(weekly_alerts.index),
            y=list(weekly_alerts.values),
            mode='lines+markers',
            name='Weekly Alerts',
            line=dict(color='#E74C3C', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Weekly Alert Frequency",
            xaxis_title="Week of Year",
            yaxis_title="Number of Alerts",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Emergency protocols
    st.markdown("### 🆘 Emergency Response Protocols")
    
    with st.expander("🔴 High Risk Event Protocol", expanded=False):
        st.markdown("""
        **Immediate Actions:**
        1. Alert all stakeholders within 5 minutes
        2. Activate emergency response team
        3. Implement protective measures for critical infrastructure
        4. Monitor continuously until event subsides
        
        **Communication Protocol:**
        - Notify operations center immediately
        - Send alerts to mobile devices
        - Update status dashboard every 15 minutes
        - Prepare situation reports for management
        """)
    
    with st.expander("🟡 Medium Risk Event Protocol", expanded=False):
        st.markdown("""
        **Standard Actions:**
        1. Monitor event progression closely
        2. Prepare contingency measures
        3. Brief relevant teams on potential impacts
        4. Increase monitoring frequency
        
        **Communication Protocol:**
        - Send standard alert notifications
        - Update status every 30 minutes
        - Prepare to escalate if conditions worsen
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
