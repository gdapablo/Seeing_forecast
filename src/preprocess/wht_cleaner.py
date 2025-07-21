# src/preprocess/wht_cleaner.py
import pandas as pd
import numpy as np
import glob
from typing import List

class WHTCleaner:
    def __init__(self, folder_path: str):
        """
        Initialize the WHTCleaner with the path to the folder containing CSV files.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing the weather data CSV files.
        """
        self.folder_path = folder_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Load and concatenate all CSV files from the folder.

        Returns
        -------
        pd.DataFrame
            Raw concatenated DataFrame from all CSV parts.
        """
        files = sorted(glob.glob(f"{self.folder_path}/*.csv"))
        data_parts = [pd.read_csv(fp) for fp in files]
        self.df = pd.concat(data_parts, ignore_index=True)
        return self.df

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the raw weather data:
        - Parse and extract datetime components
        - Convert values to float
        - Handle missing values
        - Drop columns with >30% NaNs
        - Drop remaining rows with NaNs

        Returns
        -------
        pd.DataFrame
            Cleaned and preprocessed DataFrame.
        """
        self.df['sampletime'] = pd.to_datetime(self.df['sampletime'])
        self.df['year'] = self.df['sampletime'].dt.year
        self.df['month'] = self.df['sampletime'].dt.month
        self.df['day'] = self.df['sampletime'].dt.day
        self.df['hour'] = self.df['sampletime'].dt.hour
        self.df['minute'] = self.df['sampletime'].dt.minute
        self.df['second'] = self.df['sampletime'].dt.second
        self.df.drop(columns=['sampletime'], inplace=True)

        # Replace '\\N' with -1 and convert to float32
        for col in self.df.columns[1:]:
            self.df[col] = self.df[col].replace('\\N', -1).astype('float32')

        # Handle windspeed parsing
        if 'windspeed' in self.df.columns:
            self.df['windspeed'] = pd.to_numeric(self.df['windspeed'], errors='coerce')

        # Drop columns with more than 30% NaNs
        threshold = len(self.df) * 0.3
        self.df.dropna(thresh=threshold, axis=1, inplace=True)

        # Drop rows with any remaining NaNs
        self.df.dropna(inplace=True)

        return self.df
