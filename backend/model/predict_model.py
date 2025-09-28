import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
import os

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChurnPredictorService:
    """
    A service for predicting customer churn based on various customer features.

    This class provides methods to calculate churn probability, assign risk levels,
    and identify top reasons for churn for individual customers or a batch of customers.
    It uses a simplified rule-based model for demonstration purposes.
    """
    
    def __init__(self):
        """
        Initializes the ChurnPredictorService with predefined churn probability thresholds.
        """
        self.high_churn_threshold = 0.7  # Probability threshold for 'High' churn risk
        self.medium_churn_threshold = 0.3 # Probability threshold for 'Medium' churn risk
        
    def _get_risk_level(self, probability: float) -> str:
        """
        Determines the churn risk level (High, Medium, Low) based on a given churn probability.

        Args:
            probability (float): The calculated churn probability for a customer.

        Returns:
            str: The risk level ('High', 'Medium', or 'Low').
        """
        if probability >= self.high_churn_threshold:
            return "High"
        elif probability >= self.medium_churn_threshold:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_churn_score(self, row: pd.Series) -> float:
        """
        Calculates a churn score for a single customer based on their features.

        This is a rule-based scoring mechanism where different customer attributes
        contribute to the overall churn score. A higher score indicates a higher
        likelihood of churn.

        Args:
            row (pd.Series): A pandas Series representing a single customer's data.

        Returns:
            float: The calculated churn score, normalized between 0.0 and 1.0.
        """
        score = 0.0
        
        # Days since signup: Newer customers might have higher churn risk
        if row.get('days_since_signup', 0) < 30:
            score += 0.2
        elif row.get('days_since_signup', 0) < 90:
            score += 0.1
        
        # Last login days ago: Inactive users are more likely to churn
        last_login_days = row.get('last_login_days_ago', 0)
        if last_login_days > 30:
            score += 0.3
        elif last_login_days > 14:
            score += 0.2
        elif last_login_days > 7:
            score += 0.1
        
        # Billing issues: Presence of billing issues increases churn risk
        if row.get('billing_issue_count', 0) > 0:
            score += 0.25
        
        # Support tickets: A high number of support tickets can indicate dissatisfaction
        support_tickets = row.get('support_tickets_opened', 0)
        if support_tickets > 5:
            score += 0.2
        elif support_tickets > 2:
            score += 0.1
        
        # Email engagement: Low email engagement might signal disinterest
        email_opens = row.get('email_opens_last30days', 0)
        if email_opens < 3:
            score += 0.15
        
        # Login frequency: Low login frequency indicates low product usage
        logins = row.get('number_of_logins_last30days', 0)
        if logins < 5:
            score += 0.2
        elif logins < 10:
            score += 0.1
        
        # Revenue: Lower revenue customers might be less sticky
        revenue = row.get('monthly_revenue', 0)
        if revenue < 50:
            score += 0.1
        
        # Add some randomness to the score for demo purposes to simulate model variability
        score += np.random.normal(0, 0.1)
        
        # Ensure the score is within the valid range [0.0, 1.0]
        return max(0.0, min(1.0, score))
    
    def _get_top_reasons(self, row: pd.Series, score: float) -> List[str]:
        """
        Identifies the top churn reasons for a given customer based on their features and churn score.

        This function uses a set of rules to associate specific customer behaviors or attributes
        with potential churn reasons.

        Args:
            row (pd.Series): A pandas Series representing a single customer's data.
            score (float): The calculated churn score for the customer.

        Returns:
            List[str]: A list of top 2 churn reasons. Returns ['general_risk'] if no specific reasons are found.
        """
        reasons = []
        
        # Reason: Low engagement due to infrequent logins
        if row.get('last_login_days_ago', 0) > 14:
            reasons.append('low_engagement')
        
        # Reason: Billing issues
        if row.get('billing_issue_count', 0) > 0:
            reasons.append('billing_issues')
        
        # Reason: Support issues due to high number of tickets
        if row.get('support_tickets_opened', 0) > 2:
            reasons.append('support_issues')
        
        # Reason: Low email engagement
        if row.get('email_opens_last30days', 0) < 3:
            reasons.append('low_email_engagement')
        
        # Reason: Low product usage due to infrequent logins
        if row.get('number_of_logins_last30days', 0) < 5:
            reasons.append('low_usage')
        
        # Reason: New customer, often higher churn risk in early stages
        if row.get('days_since_signup', 0) < 30:
            reasons.append('new_customer')
        
        # Reason: Low-value customer, potentially less invested in the product
        if row.get('monthly_revenue', 0) < 50:
            reasons.append('low_value_customer')
        
        # Return top 2 reasons to keep the output concise
        return reasons[:2] if reasons else ['general_risk']
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates churn predictions for a batch of customers provided in a DataFrame.

        For each customer, it calculates a churn probability, assigns a risk level,
        and identifies the top reasons contributing to their churn risk.

        Args:
            df (pd.DataFrame): A DataFrame where each row represents a customer
                                and columns are their features.

        Returns:
            pd.DataFrame: A DataFrame containing the 'user_id', 'churn_probability',
                          'risk_level', and 'top_reasons' for each customer.

        Raises:
            ValueError: If the input DataFrame does not contain a 'user_id' column.
        """
        logging.info(f"Starting batch prediction for {len(df)} customers.")
        
        if 'user_id' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'user_id' column.")
        
        results = []
        
        for index, row in df.iterrows():
            user_id = row['user_id']
            
            # Calculate churn score for the current customer
            churn_score = self._calculate_churn_score(row)
            
            # Determine the risk level based on the churn score
            risk_level = self._get_risk_level(churn_score)
            
            # Get the top reasons for the calculated churn score
            top_reasons = self._get_top_reasons(row, churn_score)
            
            results.append({
                "user_id": user_id,
                "churn_probability": round(float(churn_score), 4),
                "risk_level": risk_level,
                "top_reasons": top_reasons
            })
        
        logging.info("Batch prediction completed.")
        return pd.DataFrame(results)