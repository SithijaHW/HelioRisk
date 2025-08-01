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
from utils.ml_models import train_random_forest, make_predictions
from utils.visualizations import (
    create_overview_charts, create_time_series_chart, 
    create_correlation_heatmap, create_impact_distribution,
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
    
    try:
        data = load_data()
        
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
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not data['power_grid'].empty:
            power_events = len(data['power_grid'])
            st.markdown(f"""
            <div class="metric-card">
                <h4>⚡ Power Grid Events</h4>
                <h2>{power_events:,}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if not data['satellite'].empty:
            sat_events = len(data['satellite'])
            st.markdown(f"""
            <div class="metric-card">
                <h4>🛰️ Satellite Anomalies</h4>
                <h2>{sat_events:,}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if not data['solar_flare'].empty:
            flare_events = len(data['solar_flare'])
            st.markdown(f"""
            <div class="metric-card">
                <h4>☀️ Solar Flares</h4>
                <h2>{flare_events:,}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if not data['solar_wind'].empty:
            wind_events = len(data['solar_wind'])
            st.markdown(f"""
            <div class="metric-card">
                <h4>🌊 Solar Wind Events</h4>
                <h2>{wind_events:,}</h2>
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
    if not data['power_grid'].empty:
        fig = create_geographical_analysis(data['power_grid'])
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
            with st.spinner("Training Random Forest model..."):
                if not data['power_grid'].empty:
                    model, accuracy, feature_importance, predictions = train_random_forest(
                        data['power_grid'], n_estimators, max_depth, test_size
                    )
                    
                    st.session_state['model'] = model
                    st.session_state['accuracy'] = accuracy
                    st.session_state['feature_importance'] = feature_importance
                    st.session_state['predictions'] = predictions
                    
                    st.success(f"✅ Model trained! Accuracy: {accuracy:.2%}")
    
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
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        duration = st.number_input("Event Duration (hours)", min_value=1, max_value=24, value=5)
        region = st.selectbox("Region", ["USA", "Europe", "China", "India"])
    
    with pred_col2:
        cause = st.selectbox("Cause", ["Overload", "Hardware Fault", "Geomagnetic Disturbance"])
        hour = st.slider("Hour of Day", 0, 23, 12)
    
    with pred_col3:
        if st.button("🎯 Predict Impact Level", type="primary"):
            if 'model' in st.session_state:
                prediction = make_predictions(st.session_state['model'], duration, region, cause, hour)
                
                if prediction == 'High':
                    st.markdown(f"""
                    <div class="alert-high">
                        <h4>⚠️ Predicted Impact: HIGH RISK</h4>
                        <p>Immediate attention required. Implement emergency protocols.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prediction == 'Medium':
                    st.markdown(f"""
                    <div class="alert-medium">
                        <h4>⚡ Predicted Impact: MEDIUM RISK</h4>
                        <p>Monitor situation closely. Prepare contingency measures.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-low">
                        <h4>✅ Predicted Impact: LOW RISK</h4>
                        <p>Normal operations can continue with standard monitoring.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please train the model first!")
    
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
    st.markdown("### 🔗 Correlation Analysis")
    
    if not data['power_grid'].empty:
        fig = create_correlation_heatmap(data['power_grid'])
        st.plotly_chart(fig, use_container_width=True)
    
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
    
    # Interactive filters
    st.markdown("### 🎛️ Dynamic Filters")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        viz_type = st.selectbox(
            "Visualization Type",
            ["Scatter Plot", "3D Analysis", "Bubble Chart", "Sunburst", "Treemap"]
        )
    
    with filter_col2:
        if not data['power_grid'].empty:
            x_axis = st.selectbox(
                "X-Axis",
                ['Duration', 'Region', 'Cause', 'Impact_Level']
            )
    
    with filter_col3:
        if not data['power_grid'].empty:
            y_axis = st.selectbox(
                "Y-Axis", 
                ['Duration', 'Impact_Level', 'Cause', 'Region']
            )
    
    # Generate interactive visualization
    if not data['power_grid'].empty:
        df = data['power_grid'].copy()
        
        if viz_type == "Scatter Plot":
            if x_axis == 'Duration' and y_axis == 'Duration':
                # Create hour vs duration scatter
                df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
                df['Hour'] = df['Time'].dt.hour
                
                fig = px.scatter(
                    df,
                    x='Hour',
                    y='Duration',
                    color='Impact_Level',
                    size='Duration',
                    hover_data=['Region', 'Cause'],
                    title="Event Duration vs Hour of Day",
                    color_discrete_map={
                        'Low': '#27AE60',
                        'Medium': '#F39C12',
                        'High': '#E74C3C'
                    }
                )
                
            else:
                # Encode categorical variables
                if x_axis in ['Region', 'Cause', 'Impact_Level']:
                    df[f'{x_axis}_encoded'] = pd.Categorical(df[x_axis]).codes
                    x_col = f'{x_axis}_encoded'
                else:
                    x_col = x_axis
                
                if y_axis in ['Region', 'Cause', 'Impact_Level']:
                    df[f'{y_axis}_encoded'] = pd.Categorical(df[y_axis]).codes
                    y_col = f'{y_axis}_encoded'
                else:
                    y_col = y_axis
                
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color='Impact_Level',
                    title=f"{x_axis} vs {y_axis}",
                    color_discrete_map={
                        'Low': '#27AE60',
                        'Medium': '#F39C12',
                        'High': '#E74C3C'
                    }
                )
        
        elif viz_type == "3D Analysis":
            df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
            df['Hour'] = df['Time'].dt.hour
            
            fig = px.scatter_3d(
                df,
                x='Hour',
                y='Duration',
                z=pd.Categorical(df['Impact_Level']).codes,
                color='Impact_Level',
                title="3D Analysis: Hour vs Duration vs Impact Level",
                color_discrete_map={
                    'Low': '#27AE60',
                    'Medium': '#F39C12',
                    'High': '#E74C3C'
                }
            )
        
        elif viz_type == "Sunburst":
            fig = px.sunburst(
                df,
                path=['Region', 'Cause', 'Impact_Level'],
                title="Hierarchical View: Region > Cause > Impact Level"
            )
        
        elif viz_type == "Treemap":
            impact_counts = df.groupby(['Region', 'Impact_Level']).size().reset_index(name='Count')
            
            fig = px.treemap(
                impact_counts,
                path=['Region', 'Impact_Level'],
                values='Count',
                title="Treemap: Events by Region and Impact Level",
                color='Count',
                color_continuous_scale='Blues'
            )
        
        else:  # Bubble Chart
            df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
            df['Hour'] = df['Time'].dt.hour
            
            bubble_data = df.groupby(['Region', 'Impact_Level']).agg({
                'Duration': 'mean',
                'Hour': 'mean'
            }).reset_index()
            bubble_data['Count'] = df.groupby(['Region', 'Impact_Level']).size().values
            
            fig = px.scatter(
                bubble_data,
                x='Hour',
                y='Duration',
                size='Count',
                color='Impact_Level',
                hover_name='Region',
                title="Bubble Chart: Average Duration vs Hour (Size = Event Count)",
                color_discrete_map={
                    'Low': '#27AE60',
                    'Medium': '#F39C12',
                    'High': '#E74C3C'
                }
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
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

def alert_system_tab(data):
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("## 🚨 Real-time Alert System")
    
    # Current status overview
    st.markdown("### 🌐 Current Space Weather Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.markdown("""
        <div class="alert-low">
            <h4>🟢 Solar Activity</h4>
            <p><strong>Status:</strong> Normal</p>
            <p><strong>Last Update:</strong> 2 minutes ago</p>
        </div>
        """, unsafe_allow_html=True)
    
    with status_col2:
        st.markdown("""
        <div class="alert-medium">
            <h4>🟡 Geomagnetic Field</h4>
            <p><strong>Status:</strong> Moderate</p>
            <p><strong>Last Update:</strong> 5 minutes ago</p>
        </div>
        """, unsafe_allow_html=True)
    
    with status_col3:
        st.markdown("""
        <div class="alert-high">
            <h4>🔴 Solar Wind</h4>
            <p><strong>Status:</strong> Elevated</p>
            <p><strong>Last Update:</strong> 1 minute ago</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent alerts
    st.markdown("### 📢 Recent Alerts")
    
    if not data['power_grid'].empty:
        # Get recent high-impact events
        df = data['power_grid'].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        recent_high = df[df['Impact_Level'] == 'High'].nlargest(5, 'Date')
        
        for _, event in recent_high.iterrows():
            alert_time = event['Date'].strftime('%Y-%m-%d')
            st.markdown(f"""
            <div class="alert-high">
                <h5>⚠️ High Impact Event Alert</h5>
                <p><strong>Date:</strong> {alert_time}</p>
                <p><strong>Region:</strong> {event['Region']}</p>
                <p><strong>Cause:</strong> {event['Cause']}</p>
                <p><strong>Duration:</strong> {event['Duration']} hours</p>
            </div>
            """, unsafe_allow_html=True)
    
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
