import gradio as gr
import pandas as pd
import joblib

# Load the model (assuming it's in the same directory)
model = joblib.load("churn_model.pkl")

def predict_churn(csv_file):
    df = pd.read_csv(csv_file.name)
    preds = model.predict(df)  # adjust preprocessing if needed
    df["churn_prediction"] = preds
    return df

iface = gr.Interface(
    fn=predict_churn,
    inputs=gr.File(label="Upload CSV"),
    outputs=gr.Dataframe(label="Predictions")
)

iface.launch()