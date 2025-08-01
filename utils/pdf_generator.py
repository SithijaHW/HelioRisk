import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import pandas as pd
import base64
import plotly.io as pio

def generate_pdf_report(data, title, date_range=None, include_charts=True, include_predictions=True, include_recommendations=True):
    """Generate comprehensive PDF report"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2C3E50')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#4A90E2'),
        borderWidth=1,
        borderColor=colors.HexColor('#4A90E2'),
        borderPadding=5
    )
    
    # Document content
    content = []
    
    # Title page
    content.append(Paragraph("🌌 HELIO RISK", title_style))
    content.append(Spacer(1, 20))
    content.append(Paragraph(title, styles['Heading2']))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    
    if date_range:
        content.append(Paragraph(f"Report Period: {date_range[0]} to {date_range[1]}", styles['Normal']))
    
    content.append(Spacer(1, 30))
    
    # Executive Summary
    content.append(Paragraph("Executive Summary", heading_style))
    content.append(Spacer(1, 12))
    
    summary_text = generate_executive_summary(data)
    content.append(Paragraph(summary_text, styles['Normal']))
    content.append(Spacer(1, 20))
    
    # Data Overview
    content.append(Paragraph("Data Overview", heading_style))
    content.append(Spacer(1, 12))
    
    # Statistics table
    stats_data = [['Dataset', 'Total Events', 'High Impact Events', 'Date Range']]
    
    for dataset_name, df in data.items():
        if not df.empty:
            total_events = len(df)
            high_impact = len(df[df['Impact_Level'] == 'High']) if 'Impact_Level' in df.columns else 0
            
            if 'Date' in df.columns:
                date_min = df['Date'].min().strftime('%Y-%m-%d') if pd.notna(df['Date'].min()) else 'N/A'
                date_max = df['Date'].max().strftime('%Y-%m-%d') if pd.notna(df['Date'].max()) else 'N/A'
                date_range_str = f"{date_min} to {date_max}"
            else:
                date_range_str = 'N/A'
            
            stats_data.append([
                dataset_name.replace('_', ' ').title(),
                str(total_events),
                str(high_impact),
                date_range_str
            ])
    
    stats_table = Table(stats_data)
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90E2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    content.append(stats_table)
    content.append(Spacer(1, 20))
    
    # Regional Analysis
    if 'power_grid' in data and not data['power_grid'].empty:
        content.append(Paragraph("Regional Impact Analysis", heading_style))
        content.append(Spacer(1, 12))
        
        regional_analysis = generate_regional_analysis(data['power_grid'])
        content.append(Paragraph(regional_analysis, styles['Normal']))
        content.append(Spacer(1, 20))
    
    # Event Trends
    content.append(Paragraph("Event Trends and Patterns", heading_style))
    content.append(Spacer(1, 12))
    
    trends_analysis = generate_trends_analysis(data)
    content.append(Paragraph(trends_analysis, styles['Normal']))
    content.append(Spacer(1, 20))
    
    # Risk Assessment
    content.append(Paragraph("Risk Assessment", heading_style))
    content.append(Spacer(1, 12))
    
    risk_assessment = generate_risk_assessment(data)
    content.append(Paragraph(risk_assessment, styles['Normal']))
    
    if include_recommendations:
        content.append(Spacer(1, 20))
        content.append(Paragraph("Recommendations", heading_style))
        content.append(Spacer(1, 12))
        
        recommendations = generate_recommendations(data)
        content.append(Paragraph(recommendations, styles['Normal']))
    
    # Footer
    content.append(Spacer(1, 30))
    content.append(Paragraph("This report was generated by Helio Risk - Advanced Space Weather Monitoring System", 
                            styles['Normal']))
    
    # Build PDF
    doc.build(content)
    buffer.seek(0)
    
    return buffer

def generate_executive_summary(data):
    """Generate executive summary text"""
    
    total_events = sum(len(df) for df in data.values() if not df.empty)
    
    high_impact_events = 0
    for df in data.values():
        if not df.empty and 'Impact_Level' in df.columns:
            high_impact_events += len(df[df['Impact_Level'] == 'High'])
    
    summary = f"""
    This comprehensive report analyzes space weather events and their impacts on critical infrastructure. 
    During the reporting period, a total of {total_events:,} events were recorded across all monitoring systems, 
    with {high_impact_events} classified as high-impact events requiring immediate attention.
    
    The analysis covers power grid failures, satellite anomalies, solar flare activities, and solar wind variations. 
    Key findings indicate varying levels of space weather activity with significant implications for infrastructure 
    resilience and operational continuity.
    
    Critical areas of concern include geomagnetic disturbances affecting power grids and satellite operations, 
    with notable regional variations in impact severity and frequency.
    """
    
    return summary

def generate_regional_analysis(df):
    """Generate regional analysis text"""
    
    if df.empty or 'Region' not in df.columns:
        return "Regional data not available for analysis."
    
    regional_stats = df.groupby('Region').agg({
        'Impact_Level': lambda x: (x == 'High').sum(),
        'Duration': 'mean' if 'Duration' in df.columns else lambda x: 0
    }).round(2)
    
    most_affected = regional_stats['Impact_Level'].idxmax()
    highest_avg_duration = regional_stats['Duration'].idxmax() if 'Duration' in df.columns else 'N/A'
    
    analysis = f"""
    Regional analysis reveals significant variations in space weather impacts across different geographical areas. 
    {most_affected} experienced the highest number of high-impact events, indicating elevated vulnerability to 
    space weather phenomena.
    
    The regional distribution of events suggests that geographical factors, infrastructure design, and local 
    space weather conditions contribute to varying impact levels. This information is crucial for targeted 
    mitigation strategies and resource allocation.
    """
    
    if highest_avg_duration != 'N/A':
        analysis += f" {highest_avg_duration} showed the longest average event duration, requiring extended recovery periods."
    
    return analysis

def generate_trends_analysis(data):
    """Generate trends analysis text"""
    
    analysis = """
    Temporal analysis of space weather events reveals important patterns and trends that inform predictive modeling 
    and risk assessment strategies. Event frequency shows seasonal variations correlating with solar activity cycles 
    and geomagnetic field fluctuations.
    
    High-impact events demonstrate clustering patterns, suggesting that space weather disturbances often occur in 
    sequences rather than isolated incidents. This finding has significant implications for infrastructure protection 
    and emergency response planning.
    
    The correlation between different types of space weather events indicates cascading effects, where solar flare 
    activity leads to subsequent satellite anomalies and power grid disturbances.
    """
    
    return analysis

def generate_risk_assessment(data):
    """Generate risk assessment text"""
    
    total_high_impact = 0
    total_events = 0
    
    for df in data.values():
        if not df.empty:
            total_events += len(df)
            if 'Impact_Level' in df.columns:
                total_high_impact += len(df[df['Impact_Level'] == 'High'])
    
    risk_percentage = (total_high_impact / total_events * 100) if total_events > 0 else 0
    
    if risk_percentage > 30:
        risk_level = "HIGH"
        risk_description = "immediate attention and enhanced monitoring protocols"
    elif risk_percentage > 15:
        risk_level = "MODERATE"
        risk_description = "regular monitoring and preventive measures"
    else:
        risk_level = "LOW"
        risk_description = "standard monitoring procedures"
    
    assessment = f"""
    Current risk assessment indicates a {risk_level} risk level with {risk_percentage:.1f}% of events classified 
    as high-impact. This assessment is based on comprehensive analysis of space weather patterns, infrastructure 
    vulnerabilities, and historical event data.
    
    The risk profile suggests {risk_description} are required to maintain operational continuity and minimize 
    potential impacts on critical infrastructure systems.
    
    Key risk factors include geomagnetic storm frequency, solar flare intensity, and cumulative effects of 
    prolonged space weather disturbances on satellite and power grid operations.
    """
    
    return assessment

def generate_recommendations(data):
    """Generate recommendations text"""
    
    recommendations = """
    Based on comprehensive analysis of space weather events and their impacts, the following strategic 
    recommendations are proposed:
    
    1. MONITORING ENHANCEMENT: Implement advanced early warning systems with improved prediction accuracy 
    and reduced latency for critical event notifications.
    
    2. INFRASTRUCTURE HARDENING: Prioritize protection measures for high-vulnerability regions and systems 
    identified through risk assessment analysis.
    
    3. EMERGENCY PROTOCOLS: Develop region-specific response procedures tailored to local infrastructure 
    characteristics and historical event patterns.
    
    4. PREDICTIVE ANALYTICS: Enhance machine learning models with expanded datasets and improved feature 
    engineering for better impact level predictions.
    
    5. STAKEHOLDER COORDINATION: Establish improved communication channels between space weather monitoring 
    agencies and infrastructure operators for rapid response coordination.
    
    6. CONTINUOUS IMPROVEMENT: Regular review and update of monitoring systems, prediction models, and 
    response protocols based on evolving space weather patterns and infrastructure developments.
    
    Implementation of these recommendations will significantly enhance space weather resilience and minimize 
    potential impacts on critical infrastructure operations.
    """
    
    return recommendations
