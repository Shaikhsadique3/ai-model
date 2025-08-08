# Churnaizer API

This repository contains the FastAPI application for the Churnaizer API, designed for churn prediction and deployment on Fly.io.

## Project Structure

```
churnaizer-api/
├── app.py
├── requirements.txt
├── Dockerfile
├── fly.toml
└── .env.example
└── README.md
```

## Deployment on Fly.io

Follow these steps to deploy the Churnaizer API to Fly.io:

1.  **Login to Fly.io CLI:**

    ```bash
    flyctl auth login
    ```

2.  **Launch the application (without deploying yet):**

    This command will create a new Fly.io application and generate a `fly.toml` file. We've already provided one, so you can skip this if you're using the provided `fly.toml`.

    ```bash
    flyctl launch --name churnaizer-api --no-deploy
    ```

3.  **Set Secrets:**

    Your application requires certain environment variables for configuration. Set them using `flyctl secrets set`:

    ```bash
    flyctl secrets set AI_MODEL_PATH="path/to/your/model.pkl" OPENROUTER_API_KEY="your_openrouter_key" RESEND_API_KEY="your_resend_key"
    ```

    **Note:** Replace the placeholder values with your actual model path and API keys.

4.  **Deploy the application:**

    ```bash
    flyctl deploy
    ```

    This command will build the Docker image and deploy your application to Fly.io.

## Running Locally

To run the application locally for development or testing:

1.  **Navigate to the `churnaizer-api` directory:**

    ```bash
    cd churnaizer-api
    ```

2.  **Install dependencies:**

    It's recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Create a `.env` file:**

    Copy the `.env.example` file to `.env` and fill in your actual values:

    ```bash
    cp .env.example .env
    # Edit .env with your actual values
    ```

4.  **Run the application with Uvicorn:**

    ```bash
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    ```

    The API will be accessible at `http://localhost:8000`.

## API Endpoints

*   **GET /health**
    Returns `{"status": "ok"}`. Use this to check if the application is running.

*   **POST /api/v1/predict**
    Accepts a JSON body with user data and returns churn prediction results.

    **Request Body Example:**
    ```json
    {
        "user_id": "user123",
        "plan": "premium",
        "usage_score": 0.75,
        "support_tickets": 1,
        "email": "user@example.com"
    }
    ```

    **Response Body Example:**
    ```json
    {
        "user_id": "user123",
        "churn_probability": 0.15,
        "risk_level": "low",
        "message": "User is unlikely to churn based on current data.",
        "trigger_email": false,
        "recommended_email_tone": "neutral"
    }
    ```