import pandas as pd

class SeeingCleaner:
    """
    Class to load and preprocess seeing data from CSV.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing seeing data.

    Methods
    -------
    load_and_preprocess()
        Loads the CSV, converts date/time columns, extracts datetime components,
        drops original columns, and returns the cleaned DataFrame.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_and_preprocess(self) -> pd.DataFrame:
        """
        Load the seeing CSV and preprocess it.

        - Converts 'Date' to datetime.
        - Converts 'Time' to timedelta and extracts hour, minute, second.
        - Extracts year, month, day from 'Date'.
        - Drops original 'Date' and 'Time' columns.

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame ready for visualization or analysis.
        """
        seeing = pd.read_csv(self.filepath)
        seeing['Date'] = pd.to_datetime(seeing['Date'])
        seeing['Time'] = pd.to_timedelta(seeing['Time'])

        seeing['year'] = seeing['Date'].dt.year
        seeing['month'] = seeing['Date'].dt.month
        seeing['day'] = seeing['Date'].dt.day

        seeing['hour'] = seeing['Time'].dt.components.hours
        seeing['minute'] = seeing['Time'].dt.components.minutes
        seeing['second'] = seeing['Time'].dt.components.seconds

        seeing.drop(columns=['Date', 'Time'], inplace=True)

        return seeing
