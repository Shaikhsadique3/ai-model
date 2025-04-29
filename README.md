# AI Churn Prediction Model Deployment

This repository contains a machine learning model for predicting customer churn, built with FastAPI and scikit-learn.

## Deployment Instructions for Render.com

### Prerequisites

- A [Render.com](https://render.com) account
- Git repository with this code

### Deployment Steps

1. **Login to Render.com**
   - Create an account or log in to your existing account

2. **Create a New Web Service**
   - Click on "New" and select "Web Service"
   - Connect your GitHub/GitLab repository or use the public URL

3. **Configure the Web Service**
   - Name: Choose a name for your service (e.g., "churn-prediction-api")
   - Environment: Select "Python"
   - Region: Choose the region closest to your users
   - Branch: Select your main branch (e.g., "main" or "master")
   - Build Command: `pip install -r requirements.txt`
   - Start Command: Leave as is (Render will use the Procfile)

4. **Add Environment Variables (if needed)**
   - Click on "Environment" tab
   - Add any secret keys or configuration variables

5. **Deploy the Service**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

6. **Access Your API**
   - Once deployment is complete, you can access your API at the URL provided by Render
   - The API documentation will be available at `/docs`

## API Usage

### Authentication

All API requests require an API key that should be included in the header:

```
X-API-Key: your_api_key
```

You can get your API key from the admin dashboard at `/admin`.

### Prediction Endpoint

```
POST /api/v1/predict
```

Example request body:

```json
{
  "days_since_signup": 120,
  "monthly_revenue": 50,
  "number_of_logins_last30days": 5,
  "active_features_used": 3,
  "support_tickets_opened": 2,
  "last_payment_status": "Success",
  "subscription_plan": "Premium"
}
```

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   uvicorn app:app --reload
   ```

3. Access the API at http://localhost:8000