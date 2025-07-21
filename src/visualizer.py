import pandas as pd
import matplotlib.pyplot as plt


class WeatherVisualizer:
    """
    Class for generating weather visualizations from INT meteorological data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the visualizer with a DataFrame.

        Parameters:
        df (pd.DataFrame): Cleaned and processed INT weather dataset.
        """
        self.df = df

    def plot_monthly_means(self):
        """
        Plot mean temperature, wind speed, and humidity grouped by month.
        """
        mean_temp = self.df.groupby('month')['localairtemperature'].mean()
        mean_wind = self.df.groupby('month')['localwindspeed'].mean()
        mean_humidity = self.df.groupby('month')['localhumidity'].mean()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax = ax.ravel()

        mean_temp.plot(kind='bar', color='skyblue', ax=ax[0])
        mean_wind.plot(kind='bar', color='skyblue', ax=ax[1])
        mean_humidity.plot(kind='bar', color='skyblue', ax=ax[2])

        ax[0].set_title('Mean Temperature per Month')
        ax[1].set_title('Mean Wind Speed per Month')
        ax[2].set_title('Mean Humidity per Month')

        for i in range(3):
            ax[i].set_xlabel('Month')
        ax[0].set_ylabel('Temperature (°C)')
        ax[1].set_ylabel('Wind Speed (km/h)')
        ax[2].set_ylabel('Humidity (%)')

        plt.tight_layout()
        plt.show()

    def plot_yearly_means(self):
        """
        Plot mean temperature, wind speed, and humidity grouped by year.
        """
        mean_temp = self.df.groupby('year')['localairtemperature'].mean()
        mean_wind = self.df.groupby('year')['localwindspeed'].mean()
        mean_humidity = self.df.groupby('year')['localhumidity'].mean()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax = ax.ravel()

        mean_temp.plot(kind='bar', color='red', ax=ax[0])
        mean_wind.plot(kind='bar', color='red', ax=ax[1])
        mean_humidity.plot(kind='bar', color='red', ax=ax[2])

        ax[0].set_title('Mean Temperature per Year')
        ax[1].set_title('Mean Wind Speed per Year')
        ax[2].set_title('Mean Humidity per Year')

        for i in range(3):
            ax[i].set_xlabel('Year')
        ax[0].set_ylabel('Temperature (°C)')
        ax[1].set_ylabel('Wind Speed (km/h)')
        ax[2].set_ylabel('Humidity (%)')

        plt.tight_layout()
        plt.show()


class SeeingVisualizer:
    """
    Class for generating seeing-related visualizations.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the visualizer with the seeing DataFrame.

        Parameters:
        df (pd.DataFrame): Cleaned seeing dataset.
        """
        self.df = df

    def plot_monthly_seeing(self):
        """
        Plot mean seeing per month and the difference w.r.t. the minimum.
        """
        mean_seeing = self.df.groupby('month')['Seeing'].mean()
        diff_seeing = mean_seeing - mean_seeing.min()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        ax = ax.ravel()

        mean_seeing.plot(kind='bar', color='red', ax=ax[0])
        diff_seeing.plot(kind='bar', color='orange', ax=ax[1])

        ax[0].set_title('Mean Seeing per Month')
        ax[1].set_title('Δ Seeing vs Minimum')

        for i in range(2):
            ax[i].set_xlabel('Month')
        ax[0].set_ylabel('Seeing (arcsec)')

        plt.tight_layout()
        plt.show()

    def plot_seeing_variation_by_minute(self):
        """
        Plot the difference between max and min seeing per minute.
        """
        diff_by_minute = self.df.groupby(
            [self.df['hour'], self.df['minute']]
        )['Seeing'].apply(lambda x: x.max() - x.min())

        plt.figure(figsize=(10, 6))
        diff_by_minute.plot(marker='o', linestyle='-')
        plt.title('Difference Between Max and Min Seeing per Minute')
        plt.xlabel('Time (Hour:Minute)')
        plt.ylabel('Difference (arcsec)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
