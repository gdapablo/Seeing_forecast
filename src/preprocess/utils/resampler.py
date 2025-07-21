# src/utils/resampler.py
import pandas as pd

def resample_weather_data(df: pd.DataFrame, freq: str = "10T") -> pd.DataFrame:
    """
    Resample weather data to a specified time frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned weather data containing 'year', 'month', 'day', 'hour', 'minute', 'second' columns.
    freq : str, optional
        Frequency for resampling (default is '10T' = 10 minutes).

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with original time components restored and datetime column removed.
    """
    df = df.copy()
    df['Datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
    df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)

    # Resample based on datetime column
    resampled_df = df.resample(freq, on='Datetime').mean()
    resampled_df.reset_index(inplace=True)

    # Recover time components
    resampled_df['year'] = resampled_df['Datetime'].dt.year
    resampled_df['month'] = resampled_df['Datetime'].dt.month
    resampled_df['day'] = resampled_df['Datetime'].dt.day
    resampled_df['hour'] = resampled_df['Datetime'].dt.hour
    resampled_df['minute'] = resampled_df['Datetime'].dt.minute

    resampled_df.drop(columns=['Datetime'], inplace=True)
    return resampled_df
