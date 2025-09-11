import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChurnPredictorService:
    """Simplified churn prediction service for demo purposes."""
    
    def __init__(self):
        self.high_churn_threshold = 0.7
        self.medium_churn_threshold = 0.3
        
    def _get_risk_level(self, probability: float) -> str:
        """Determine risk level based on churn probability."""
        if probability >= self.high_churn_threshold:
            return "High"
        elif probability >= self.medium_churn_threshold:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_churn_score(self, row: pd.Series) -> float:
        """Calculate churn score based on customer features."""
        score = 0.0
        
        # Days since signup (newer customers more likely to churn)
        if row.get('days_since_signup', 0) < 30:
            score += 0.2
        elif row.get('days_since_signup', 0) < 90:
            score += 0.1
        
        # Last login days ago (inactive users more likely to churn)
        last_login_days = row.get('last_login_days_ago', 0)
        if last_login_days > 30:
            score += 0.3
        elif last_login_days > 14:
            score += 0.2
        elif last_login_days > 7:
            score += 0.1
        
        # Billing issues
        if row.get('billing_issue_count', 0) > 0:
            score += 0.25
        
        # Support tickets (high number indicates problems)
        support_tickets = row.get('support_tickets_opened', 0)
        if support_tickets > 5:
            score += 0.2
        elif support_tickets > 2:
            score += 0.1
        
        # Email engagement
        email_opens = row.get('email_opens_last30days', 0)
        if email_opens < 3:
            score += 0.15
        
        # Login frequency
        logins = row.get('number_of_logins_last30days', 0)
        if logins < 5:
            score += 0.2
        elif logins < 10:
            score += 0.1
        
        # Revenue (lower revenue customers more likely to churn)
        revenue = row.get('monthly_revenue', 0)
        if revenue < 50:
            score += 0.1
        
        # Add some randomness for demo
        score += np.random.normal(0, 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _get_top_reasons(self, row: pd.Series, score: float) -> List[str]:
        """Get top churn reasons for a customer."""
        reasons = []
        
        if row.get('last_login_days_ago', 0) > 14:
            reasons.append('low_engagement')
        
        if row.get('billing_issue_count', 0) > 0:
            reasons.append('billing_issues')
        
        if row.get('support_tickets_opened', 0) > 2:
            reasons.append('support_issues')
        
        if row.get('email_opens_last30days', 0) < 3:
            reasons.append('low_email_engagement')
        
        if row.get('number_of_logins_last30days', 0) < 5:
            reasons.append('low_usage')
        
        if row.get('days_since_signup', 0) < 30:
            reasons.append('new_customer')
        
        if row.get('monthly_revenue', 0) < 50:
            reasons.append('low_value_customer')
        
        # Return top 2 reasons
        return reasons[:2] if reasons else ['general_risk']
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate churn predictions for a batch of customers."""
        logging.info(f"Starting batch prediction for {len(df)} customers.")
        
        if 'user_id' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'user_id' column.")
        
        results = []
        
        for index, row in df.iterrows():
            user_id = row['user_id']
            
            # Calculate churn score
            churn_score = self._calculate_churn_score(row)
            
            # Determine risk level
            risk_level = self._get_risk_level(churn_score)
            
            # Get top reasons
            top_reasons = self._get_top_reasons(row, churn_score)
            
            results.append({
                "user_id": user_id,
                "churn_probability": round(float(churn_score), 4),
                "risk_level": risk_level,
                "top_reasons": top_reasons
            })
        
        logging.info("Batch prediction completed.")
        return pd.DataFrame(results)

# For backward compatibility
def predict_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Standalone function for churn prediction."""
    predictor = ChurnPredictorService()
    return predictor.predict_batch(df)