from src.process_csv import process_csv
import pandas as pd # You'll need pandas to read the processed CSV for printing

if __name__ == '__main__':
    print("Starting CSV processing...")
    process_csv("input.csv", "my_processed_data.csv") # Call the function

    print("\nProcessing complete.")

    try:
        processed_df = pd.read_csv("processed_data.csv")
        print("\nFirst 10 rows of processed_data.csv:")
        print(processed_df.head(10))
    except FileNotFoundError:
        print("Error: processed_data.csv not found.")

    print("\nSaved files:")
    print(" - processed_data.csv")
    print(" - stats_summary.json")
    print(" - log.txt")