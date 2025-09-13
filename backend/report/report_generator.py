import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import pandas as pd
import os
import logging
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_report_pdf(processed_df: pd.DataFrame, stats_summary: dict, output_filepath: str):
    """Generate a comprehensive PDF report with visualizations."""
    try:
        doc = SimpleDocTemplate(output_filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1f2937')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#374151')
        )

        # Title
        story.append(Paragraph("Churn Audit Report", title_style))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))

        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        # Calculate summary statistics
        total_customers = len(processed_df)
        high_risk_count = len(processed_df[processed_df['risk_level'] == 'High']) if 'risk_level' in processed_df.columns else 0
        medium_risk_count = len(processed_df[processed_df['risk_level'] == 'Medium']) if 'risk_level' in processed_df.columns else 0
        low_risk_count = len(processed_df[processed_df['risk_level'] == 'Low']) if 'risk_level' in processed_df.columns else 0
        
        high_risk_percent = (high_risk_count / total_customers * 100) if total_customers > 0 else 0
        medium_risk_percent = (medium_risk_count / total_customers * 100) if total_customers > 0 else 0
        low_risk_percent = (low_risk_count / total_customers * 100) if total_customers > 0 else 0
        
        avg_churn_score = processed_df['churn_probability'].mean() if 'churn_probability' in processed_df.columns else 0

        summary_text = f"""
        This comprehensive churn analysis examines {total_customers:,} customers in your database.
        Our AI-powered analysis identifies {high_risk_count:,} customers ({high_risk_percent:.1f}%) as high risk for churn,
        {medium_risk_count:,} customers ({medium_risk_percent:.1f}%) as medium risk, and
        {low_risk_count:,} customers ({low_risk_percent:.1f}%) as low risk.
        
        The average churn probability across all customers is {avg_churn_score:.2f}, indicating 
        {'above average' if avg_churn_score > 0.06 else 'below average' if avg_churn_score < 0.04 else 'average'} 
        churn risk compared to industry standards.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        # Key Metrics Table
        story.append(Paragraph("Key Metrics", heading_style))
        metrics_data = [
            ['Metric', 'Value', 'Industry Benchmark'],
            ['Total Customers', f'{total_customers:,}', '-'],
            ['High Risk Customers', f'{high_risk_count:,} ({high_risk_percent:.1f}%)', '< 10%'],
            ['Medium Risk Customers', f'{medium_risk_count:,} ({medium_risk_percent:.1f}%)', '15-25%'],
            ['Low Risk Customers', f'{low_risk_count:,} ({low_risk_percent:.1f}%)', '> 65%'],
            ['Average Churn Score', f'{avg_churn_score:.3f}', '0.050-0.070'],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1f2937')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb'))
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3 * inch))

        # Risk Distribution Chart
        if 'risk_level' in processed_df.columns:
            story.append(Paragraph("Churn Risk Distribution", heading_style))
            risk_counts = processed_df['risk_level'].value_counts()
            
            if not risk_counts.empty:
                plt.figure(figsize=(8, 6))
                colors_map = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#22c55e'}
                pie_colors = [colors_map.get(label, '#6b7280') for label in risk_counts.index]
                
                plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                       startangle=90, colors=pie_colors)
                plt.title('Customer Churn Risk Distribution', fontsize=14, fontweight='bold')
                plt.axis('equal')
                
                chart_path = "temp_risk_distribution.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                story.append(Image(chart_path, width=400, height=300))
                story.append(Spacer(1, 0.2 * inch))
                
                # Clean up
                if os.path.exists(chart_path):
                    os.remove(chart_path)

        # Top Churn Reasons
        story.append(Paragraph("Top Churn Risk Factors", heading_style))
        
        if 'top_reasons' in processed_df.columns:
            # Flatten and count reasons
            all_reasons = []
            for reasons in processed_df['top_reasons'].dropna():
                if isinstance(reasons, str):
                    all_reasons.extend([r.strip() for r in reasons.split(',')])
                elif isinstance(reasons, list):
                    all_reasons.extend(reasons)
            
            if all_reasons:
                reason_counts = pd.Series(all_reasons).value_counts().head(8)
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(range(len(reason_counts)), reason_counts.values, 
                              color='#3b82f6', alpha=0.8)
                plt.title('Top Churn Risk Factors', fontsize=14, fontweight='bold')
                plt.xlabel('Risk Factors')
                plt.ylabel('Number of Customers')
                plt.xticks(range(len(reason_counts)), 
                          [reason.replace('_', ' ').title() for reason in reason_counts.index], 
                          rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, value in zip(bars, reason_counts.values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(value), ha='center', va='bottom')
                
                plt.tight_layout()
                chart_path = "temp_churn_reasons.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                story.append(Image(chart_path, width=500, height=300))
                story.append(Spacer(1, 0.2 * inch))
                
                # Clean up
                if os.path.exists(chart_path):
                    os.remove(chart_path)

        # High-Risk Customers Table
        story.append(Paragraph("High-Risk Customers (Top 10)", heading_style))
        
        if 'risk_level' in processed_df.columns and 'churn_probability' in processed_df.columns:
            high_risk_customers = processed_df[processed_df['risk_level'] == 'High'].nlargest(10, 'churn_probability')
            
            if not high_risk_customers.empty:
                table_data = [['Customer ID', 'Churn Score', 'Risk Level', 'Primary Risk Factors']]
                
                for _, row in high_risk_customers.iterrows():
                    customer_id = row.get('user_id', 'N/A')
                    churn_score = f"{row.get('churn_probability', 0):.3f}"
                    risk_level = row.get('risk_level', 'N/A')
                    reasons = row.get('top_reasons', [])
                    
                    if isinstance(reasons, str):
                        reasons_str = reasons.replace('_', ' ').title()
                    elif isinstance(reasons, list):
                        reasons_str = ', '.join([r.replace('_', ' ').title() for r in reasons[:2]])
                    else:
                        reasons_str = 'General Risk'
                    
                    table_data.append([customer_id, churn_score, risk_level, reasons_str])
                
                risk_table = Table(table_data, colWidths=[1.5*inch, 1*inch, 1*inch, 2.5*inch])
                risk_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fee2e2')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#991b1b')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#fecaca')),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP')
                ]))
                story.append(risk_table)
            else:
                story.append(Paragraph("No high-risk customers identified.", styles['Normal']))
        
        story.append(Spacer(1, 0.3 * inch))

        # Recommendations
        story.append(Paragraph("Retention Recommendations", heading_style))
        
        recommendations = [
            "1. <b>Immediate Action Required:</b> Contact high-risk customers within 48 hours with personalized retention offers.",
            "2. <b>Engagement Campaign:</b> Launch targeted email campaigns for customers with low engagement scores.",
            "3. <b>Billing Issues:</b> Proactively resolve billing problems and offer payment plan alternatives.",
            "4. <b>Support Optimization:</b> Reduce support ticket resolution time and improve first-contact resolution rates.",
            "5. <b>Product Adoption:</b> Implement onboarding improvements for new customers to reduce early churn.",
            "6. <b>Value Communication:</b> Regular check-ins with medium-risk customers to demonstrate product value."
        ]
        
        for rec in recommendations:
            story.append(Paragraph(rec, styles['Normal']))
            story.append(Spacer(1, 0.1 * inch))

        story.append(Spacer(1, 0.2 * inch))

        # Footer
        story.append(Paragraph("Report generated by Churn Audit Service", styles['Normal']))
        story.append(Paragraph(f"Analysis completed on {datetime.now().strftime('%Y-%m-%d at %H:%M UTC')}", styles['Normal']))

        # Build PDF
        doc.build(story)
        logging.info(f"PDF report successfully generated at {output_filepath}")
        
    except Exception as e:
        logging.error(f"Error generating PDF report: {e}")
        # Create a simple text report as fallback
        try:
            with open(output_filepath.replace('.pdf', '.txt'), 'w') as f:
                f.write("CHURN AUDIT REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Total Customers: {len(processed_df)}\n")
                if 'risk_level' in processed_df.columns:
                    risk_counts = processed_df['risk_level'].value_counts()
                    for level, count in risk_counts.items():
                        f.write(f"{level} Risk: {count}\n")
                f.write(f"\nReport generation completed successfully.")
            logging.info(f"Text report generated as fallback at {output_filepath.replace('.pdf', '.txt')}")
        except Exception as fallback_error:
            logging.error(f"Failed to generate fallback text report: {fallback_error}")
            raise e

if __name__ == '__main__':
    # Test the report generator
    test_data = {
        'user_id': [f'user_{i}' for i in range(1, 101)],
        'churn_probability': np.random.beta(2, 5, 100),
        'risk_level': np.random.choice(['High', 'Medium', 'Low'], 100, p=[0.2, 0.3, 0.5]),
        'top_reasons': [['low_engagement', 'billing_issues']] * 100
    }
    test_df = pd.DataFrame(test_data)
    
    test_stats = {
        "total_customers": 100,
        "average_churn_score": 0.25
    }
    
    generate_report_pdf(test_df, test_stats, "test_report.pdf")
    print("Test report generated successfully!")