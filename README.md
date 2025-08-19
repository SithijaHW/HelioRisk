# Helio Risk - Space Weather Dashboard

## Overview

Helio Risk is a comprehensive space weather monitoring and analysis dashboard built with Streamlit. The application provides real-time visualization and machine learning-powered predictions for space weather events including solar flares, satellite anomalies, power grid failures, and solar wind data. The system aims to help organizations understand and mitigate risks associated with space weather phenomena through interactive dashboards, educational content, and automated PDF reporting capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with responsive design
- **Styling**: Custom CSS with light theme and professional styling using gradient backgrounds and metric cards
- **Layout**: Wide layout with expandable sidebar navigation
- **Visualization Engine**: Plotly for interactive charts and graphs including time series, correlation heatmaps, and geographical analysis

### Backend Architecture
- **Data Processing**: Modular utility-based architecture with separate modules for data loading, machine learning, visualizations, PDF generation, and educational content
- **Machine Learning**: Scikit-learn based predictive models using Random Forest classification for risk assessment
- **Data Pipeline**: CSV-based data ingestion with preprocessing and cleaning capabilities
- **Report Generation**: ReportLab-based PDF generation system for comprehensive reports

### Data Storage Solutions
- **Primary Storage**: CSV file-based data storage for space weather datasets
- **Data Types**: 
  - Power grid failure data
  - Satellite anomaly records
  - Solar flare measurements
  - Solar wind parameters
- **Data Processing**: Pandas-based data manipulation with datetime handling and feature engineering

### Key Components
- **Data Loader Module**: Handles CSV file loading and preprocessing with error handling
- **ML Models Module**: Provides feature preparation, model training, and prediction capabilities
- **Visualization Module**: Creates interactive charts for overview, time series, correlations, and geographical analysis
- **PDF Generator**: Generates comprehensive reports with charts and recommendations
- **Educational Content**: Provides scientific explanations about space weather phenomena

### Design Patterns
- **Modular Architecture**: Clear separation of concerns with dedicated utility modules
- **Error Handling**: Comprehensive error handling with graceful degradation
- **Feature Engineering**: Automated datetime feature extraction and categorical encoding
- **Data Validation**: Input validation and data type conversion with error recovery

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for dashboard interface
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Plotly**: Interactive visualization library for charts and graphs

### Machine Learning
- **Scikit-learn**: Machine learning algorithms including Random Forest, preprocessing utilities, and model evaluation metrics

### PDF Generation
- **ReportLab**: PDF document generation with support for charts, tables, and formatted text

### Data Processing
- **Datetime**: Python standard library for date and time handling
- **OS**: File system operations for CSV data loading
- **Warnings**: Error suppression for cleaner output

### Visualization Components
- **Plotly Express**: High-level plotting interface
- **Plotly Graph Objects**: Low-level plotting for custom visualizations
- **Plotly Subplots**: Multi-panel chart creation
- **Plotly IO**: Chart export and image generation capabilities

### File Dependencies
- **CSV Data Sources**: 
  - Power grid failures dataset
  - Satellite anomalies dataset
  - Solar flare data
  - Solar wind measurements
- **Asset Directory**: Organized under `attached_assets/` folder structure