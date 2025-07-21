import argparse
from src.preprocess.WHT_cleaner import WHTCleaner
from src.preprocess.INT_cleaner import INTCleaner
from src.preprocess.seeing_cleaner import SeeingCleaner

def process_wht():
    print("Processing WHT weather data...")
    cleaner = WHTCleaner(filepath_pattern="data/WHT_data/WHT_weather_set_parts_*")
    df_raw = cleaner.load_data()
    df_clean = cleaner.preprocess(df_raw)
    df_resampled = cleaner.resample(df_clean)
    print(df_resampled.head())

def process_int():
    print("Processing INT weather data...")
    cleaner = INTCleaner(filepath_pattern="data/INT_data/INT_weather_set_parts_*")
    df_raw = cleaner.load_data()
    df_clean = cleaner.preprocess(df_raw)
    print(df_clean.head())

def process_seeing():
    print("Processing seeing data...")
    cleaner = SeeingCleaner(filepath="data/seeing_data.csv")
    df_raw = cleaner.load_data()
    df_clean = cleaner.preprocess()
    print(df_clean.head())

def main():
    parser = argparse.ArgumentParser(description="Preprocessing pipeline for weather and seeing data.")
    parser.add_argument('--process', type=str, required=True,
                        choices=['WHT', 'INT', 'seeing'],
                        help="Specify which dataset to preprocess: 'WHT', 'INT', or 'seeing'")

    args = parser.parse_args()

    if args.process == 'WHT':
        process_wht()
    elif args.process == 'INT':
        process_int()
    elif args.process == 'seeing':
        process_seeing()

if __name__ == "__main__":
    main()
