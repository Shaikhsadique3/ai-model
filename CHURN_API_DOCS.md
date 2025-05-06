# Enhanced Churn Prediction API Documentation

## Overview

The enhanced Churn Prediction API now provides more detailed insights beyond simple churn prediction. The API now includes:

1. **Estimated Time to Churn** - Predicts how many days until a customer is likely to churn
2. **Detailed Churn Reasons** - Provides specific factors contributing to churn risk

## Authentication

All API requests require an API key that should be included in the header:

```
X-API-Key: your_api_key
```

## Endpoints

### Predict Churn

**URL**: `/predict`

**Method**: `POST`

**Request Body**:

```json
{
  "days_since_signup": 180,
  "monthly_revenue": 99,
  "subscription_plan": "Pro",
  "number_of_logins_last30days": 3,
  "active_features_used": 2,
  "support_tickets_opened": 4,
  "last_payment_status": "Failed"
}
```

**Response**:

```json
{
  "churn_prediction": true,
  "churn_probability": 0.85,
  "estimated_days_to_churn": 15,
  "churn_reasons": [
    {
      "factor": "Low engagement",
      "description": "Customer logged in less than 5 times in the last 30 days",
      "impact": "high"
    },
    {
      "factor": "Limited feature usage",
      "description": "Customer only using 2 features",
      "impact": "medium"
    },
    {
      "factor": "High support needs",
      "description": "Customer opened 4 support tickets",
      "impact": "high"
    },
    {
      "factor": "Payment issues",
      "description": "Last payment attempt failed",
      "impact": "critical"
    }
  ],
  "message": "High risk of churn"
}
```

## Understanding the Response

### Churn Prediction

A boolean value indicating whether the customer is predicted to churn (`true`) or not (`false`).

### Churn Probability

A float value between 0 and 1 representing the probability of churn. Higher values indicate higher risk.

### Estimated Days to Churn

An estimate of how many days until the customer is likely to churn. This is calculated based on:

- The churn probability (higher probability = fewer days)
- Customer engagement level (login frequency)
- Payment status (failed payments accelerate churn)

### Churn Reasons

An array of factors contributing to the churn risk, each containing:

- **factor**: The category of the risk factor
- **description**: A detailed explanation of the specific issue
- **impact**: The severity level (medium, high, critical)

## Example Use Cases

### Proactive Customer Retention

Use the estimated days to churn to prioritize which customers to reach out to first. Customers with fewer days to churn should be contacted more urgently.

### Targeted Interventions

Use the churn reasons to create personalized retention strategies:

- For customers with "Low engagement", send re-engagement emails
- For customers with "Limited feature usage", offer product training
- For customers with "Payment issues", provide billing support

## Health Check

**URL**: `/health`

**Method**: `GET`

**Response**:

```json
{
  "status": "healthy",
  "model": "loaded"
}
```

Use this endpoint to verify that the API is running correctly.