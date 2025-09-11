import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_report_pdf(processed_df: pd.DataFrame, stats_summary: dict, output_filepath: str):
    doc = SimpleDocTemplate(output_filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Churnaizer Churn Audit Report", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['h2']))
    summary_text = f"""
    This report provides an in-depth analysis of customer churn risk based on the provided data.
    Out of {stats_summary.get('total_customers', 0)} customers,
    {stats_summary.get('churn_distribution', {}).get('high_risk_percent', 0):.2f}% are identified as high risk,
    {stats_summary.get('churn_distribution', {}).get('medium_risk_percent', 0):.2f}% as medium risk, and
    {stats_summary.get('churn_distribution', {}).get('low_risk_percent', 0):.2f}% as low risk.
    The average churn score across all customers is {stats_summary.get('churn_distribution', {}).get('average_churn_score', 0):.2f}.
    """
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Churn Risk Distribution Chart
    story.append(Paragraph("Churn Risk Distribution", styles['h2']))
    if 'risk_level' in processed_df.columns:
        risk_counts = processed_df['risk_level'].value_counts()
        if not risk_counts.empty:
            plt.figure(figsize=(6, 4))
            plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title('Churn Risk Distribution')
            plt.axis('equal')
            chart_path = "churn_risk_distribution.png"
            plt.savefig(chart_path)
            plt.close()
            story.append(Image(chart_path, width=400, height=300))
            story.append(Spacer(1, 0.1 * inch))
            os.remove(chart_path) # Clean up chart image
        else:
            story.append(Paragraph("No churn risk data available for charting.", styles['Normal']))
    else:
        story.append(Paragraph("Churn risk level column not found in processed data.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Top Churn Reasons Chart
    story.append(Paragraph("Top Churn Reasons", styles['h2']))
    if 'reason' in processed_df.columns:
        # Flatten the list of reasons and count occurrences
        all_reasons = []
        for reasons_str in processed_df['reason'].dropna():
            all_reasons.extend([r.strip() for r in reasons_str.split(',')])
        
        if all_reasons:
            reason_counts = pd.Series(all_reasons).value_counts().head(5) # Top 5 reasons
            if not reason_counts.empty:
                plt.figure(figsize=(8, 5))
                reason_counts.plot(kind='bar')
                plt.title('Top 5 Churn Reasons')
                plt.xlabel('Reason')
                plt.ylabel('Number of Customers')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                chart_path = "top_churn_reasons.png"
                plt.savefig(chart_path)
                plt.close()
                story.append(Image(chart_path, width=500, height=350))
                story.append(Spacer(1, 0.1 * inch))
                os.remove(chart_path) # Clean up chart image
            else:
                story.append(Paragraph("No churn reasons data available for charting.", styles['Normal']))
        else:
            story.append(Paragraph("No churn reasons data available for charting.", styles['Normal']))
    else:
        story.append(Paragraph("Churn reason column not found in processed data.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Risky Users Table (Top 10)
    story.append(Paragraph("High-Risk Customers (Top 10)", styles['h2']))
    if 'risk_level' in processed_df.columns and 'user_id_masked' in processed_df.columns and 'churn_score' in processed_df.columns:
        high_risk_users = processed_df[processed_df['risk_level'] == 'High'].sort_values(by='churn_score', ascending=False).head(10)
        if not high_risk_users.empty:
            from reportlab.platypus import Table, TableStyle
            from reportlab.lib import colors

            data = [['Masked User ID', 'Churn Score', 'Risk Level', 'Reason']]
            for index, row in high_risk_users.iterrows():
                data.append([
                    row.get('user_id_masked', 'N/A'),
                    f"{row.get('churn_score', 0):.2f}",
                    row.get('risk_level', 'N/A'),
                    row.get('reason', 'N/A')
                ])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        else:
            story.append(Paragraph("No high-risk customers identified.", styles['Normal']))
    else:
        story.append(Paragraph("Required columns for risky users table not found.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Build PDF
    try:
        doc.build(story)
        logging.info(f"PDF report successfully generated at {output_filepath}")
    except Exception as e:
        logging.error(f"Error building PDF: {e}")
        raise

if __name__ == '__main__':
    # Example usage (for testing purposes)
    # Create a dummy DataFrame and stats_summary for testing
    dummy_data = {
        'user_id': [f'user_{i}' for i in range(1, 101)],
        'user_id_masked': [f'masked_{i}' for i in range(1, 101)],
        'churn_score': [i/100 for i in range(100, 0, -1)],
        'risk_level': ['High'] * 20 + ['Medium'] * 30 + ['Low'] * 50,
        'reason': ['High Usage Cost'] * 10 + ['Poor Support'] * 10 + ['Feature Missing'] * 10 + ['Competitor Offer'] * 10 + ['N/A'] * 60
    }
    dummy_df = pd.DataFrame(dummy_data)

    dummy_stats = {
        "total_customers": 100,
        "active_customers": 80,
        "failed_billing_customers": 5,
        "average_revenue": 150.75,
        "avg_login_gap": 7.2,
        "churn_distribution": {
            "high_risk_percent": 20.0,
            "medium_risk_percent": 30.0,
            "low_risk_percent": 50.0,
            "average_churn_score": 0.55
        }
    }

    output_test_path = "test_churn_report.pdf"
    generate_report_pdf(dummy_df, dummy_stats, output_test_path)
    print(f"Test PDF generated at {output_test_path}")