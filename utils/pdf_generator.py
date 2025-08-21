# pdf_generator.py
import io
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ---------------- PDF Generator ---------------- #

def generate_pdf_report(filtered_data, report_title, date_range=None,
                        include_charts=True, include_predictions=True,
                        include_recommendations=True):
    """
    Generate a polished, professional PDF report with structured sections,
    stats, charts, and narrative insights.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=60, bottomMargin=40)

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        name="TitleStyle", fontSize=22, alignment=1,
        spaceAfter=20, textColor=colors.HexColor("#1B2631"),
        fontName="Helvetica-Bold"
    )
    subtitle_style = ParagraphStyle(
        name="SubtitleStyle", fontSize=14, alignment=1,
        spaceAfter=20, textColor=colors.HexColor("#566573"),
    )
    header_style = ParagraphStyle(
        name="HeaderStyle", fontSize=16, spaceAfter=12,
        textColor=colors.HexColor("#154360"),
        fontName="Helvetica-Bold"
    )
    section_style = ParagraphStyle(
        name="SectionStyle", fontSize=12, leading=16,
        spaceAfter=8, textColor=colors.HexColor("#212F3D")
    )
    bullet_style = ParagraphStyle(
        name="BulletStyle", fontSize=11, leftIndent=20,
        bulletIndent=10, spaceAfter=6
    )

    elements = []

    # ---------------- Cover Page ---------------- #
    elements.append(Spacer(1, 100))
    elements.append(Paragraph(report_title, title_style))
    elements.append(Paragraph("HelioRisk: Space Weather Monitoring & Risk Assessment", subtitle_style))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", section_style))
    if date_range:
        elements.append(Paragraph(f"Report Date Range: {date_range[0]} to {date_range[1]}", section_style))
    elements.append(Spacer(1, 200))
    elements.append(Paragraph("Prepared by HelioRisk Dashboard", subtitle_style))
    elements.append(PageBreak())

    # ---------------- Dataset Sections ---------------- #
    for dataset_name, df in filtered_data.items():
        if df.empty:
            continue

        # Section Title
        elements.append(Paragraph(dataset_name.replace("_", " ").title(), header_style))
        elements.append(Spacer(1, 10))

        # Summary Table
        total_events = len(df)
        stats_data = [["Metric", "Value"]]

        if "Impact_Level" in df.columns:
            impact_dist = df["Impact_Level"].value_counts().to_dict()
            stats_data.append(["Impact Distribution", str(impact_dist)])
        if "Duration" in df.columns:
            stats_data.append(["Average Duration", f"{df['Duration'].mean():.2f} hrs"])
            stats_data.append(["Max Duration", f"{df['Duration'].max():.2f} hrs"])
        stats_data.append(["Total Events", total_events])

        table = Table(stats_data, hAlign="LEFT", colWidths=[200, 250])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F618D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#EBF5FB")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        # Narrative insights
        insights = []
        if "Impact_Level" in df.columns:
            insights.append(f"Most events were classified as **{df['Impact_Level'].mode()[0]} impact**.")
        if "Region" in df.columns:
            insights.append(f"The most affected region was **{df['Region'].mode()[0]}**.")
        if "Cause" in df.columns:
            insights.append(f"Frequent cause identified: **{df['Cause'].mode()[0]}**.")
        if not insights:
            insights.append("This dataset shows consistent patterns without major anomalies.")

        elements.append(Paragraph("Key Insights:", section_style))
        for i in insights:
            elements.append(Paragraph(i, bullet_style))

        elements.append(Spacer(1, 12))

        # Chart (optional)
        if include_charts:
            try:
                plt.figure(figsize=(4, 3))
                if "Impact_Level" in df.columns:
                    df["Impact_Level"].value_counts().plot(kind="bar", color=["#27AE60", "#F39C12", "#E74C3C"])
                    plt.title("Impact Level Distribution")
                    plt.ylabel("Count")
                else:
                    df.iloc[:, 1].value_counts().plot(kind="bar", color="#5DADE2")
                    plt.title(f"Distribution of {df.columns[1]}")
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format="PNG", bbox_inches="tight")
                plt.close()
                img_buf.seek(0)
                elements.append(Image(img_buf, width=350, height=250))
                elements.append(Spacer(1, 20))
            except Exception as e:
                elements.append(Paragraph(f"(Chart generation failed: {str(e)})", section_style))

        elements.append(PageBreak())

    # ---------------- ML Predictions ---------------- #
    if include_predictions:
        elements.append(Paragraph("Machine Learning Predictions", header_style))
        elements.append(Paragraph(
            "Our trained Random Forest models provide predictions of space weather impact "
            "levels for the next 72 hours. These predictive insights help operators anticipate "
            "high-risk events and implement preventive measures.",
            section_style
        ))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(
            "Use Case: Power grids can apply early load-balancing, while satellite operators "
            "can prepare shielding during predicted CME activity.",
            bullet_style
        ))
        elements.append(PageBreak())

    # ---------------- Recommendations ---------------- #
    if include_recommendations:
        elements.append(Paragraph("Recommendations", header_style))
        recs = [
            "✔ Strengthen satellite shielding during periods of strong CME activity.",
            "✔ Develop contingency protocols for power grid operations in high-risk regions.",
            "✔ Expand monitoring systems to track GPS disruptions proactively.",
            "✔ Continue retraining ML models with new real-time datasets for accuracy improvements."
        ]
        for r in recs:
            elements.append(Paragraph(r, bullet_style))

    # ---------------- Build PDF ---------------- #
    doc.build(elements)
    buffer.seek(0)
    return buffer
