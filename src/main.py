# src/main.py
from preprocess.wht_cleaner import INTCleaner
from preprocess.utils.resampler import resample_weather_data

def main():
    """
    Main script to run preprocessing and resampling on WHT weather dataset.

    - Loads data from CSV files
    - Cleans and formats it
    - Resamples to 10-minute intervals
    - Saves the result as a CSV
    """
    # Load and preprocess WHT
    cleaner = INTCleaner(folder_path='../data/INT_data')
    df_raw = cleaner.load_data()
    df_clean = cleaner.preprocess()

    # Resample data
    df_resampled = resample_weather_data(df_clean)

    # Save to disk
    df_resampled.to_csv('../data/WHT_cleaned_resampled.csv', index=False)
    print("✅ Resampled data saved.")

if __name__ == "__main__":
    main()
