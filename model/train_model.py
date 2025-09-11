from src.process_csv import process_csv
import pandas as pd # You'll need pandas to read the processed CSV for printing

if __name__ == '__main__':
    print("Starting CSV processing...")
    process_csv("input.csv", "processed_with_predictions.csv") # Call the function

    print("\nProcessing complete.")

    try:
        processed_df = pd.read_csv("processed_with_predictions.csv")
        print("\nFirst 10 rows of processed_with_predictions.csv:")
        print(processed_df.head(10))
    except FileNotFoundError:
        print("Error: processed_with_predictions.csv not found.")

    print("\nSaved files:")
    print(" - processed_with_predictions.csv")
    print(" - stats_summary.json")
    print(" - log.txt")