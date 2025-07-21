import pandas as pd

class SeeingCleaner:
    def __init__(self, filepath: str):
        """
        Initialize the SeeingCleaner with the path to the CSV file.

        Parameters
        ----------
        filepath : str
            Path to the seeing data CSV file.
        """
        self.filepath = filepath
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the seeing data from a CSV file.

        Returns
        -------
        pd.DataFrame
            Raw DataFrame loaded from the file.
        """
        self.df = pd.read_csv(self.filepath)
        return self.df

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the seeing data:
        - Convert date and time to datetime
        - Extract year, month, day, hour, minute, second
        - Drop original date and time columns

        Returns
        -------
        pd.DataFrame
            Cleaned and time-structured DataFrame.
        """
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Time'] = pd.to_datetime(self.df['Time'])

        self.df['year'] = self.df['Date'].dt.year
        self.df['month'] = self.df['Date'].dt.month
        self.df['day'] = self.df['Date'].dt.day

        self.df['hour'] = self.df['Time'].dt.hour
        self.df['minute'] = self.df['Time'].dt.minute
        self.df['second'] = self.df['Time'].dt.second

        self.df.drop(columns=['Date', 'Time'], inplace=True)

        return self.df
