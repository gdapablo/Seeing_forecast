# src/visualizer.py

import matplotlib.pyplot as plt
import os

class WeatherVisualizer:
    """
    A class for creating weather-related visualizations.

    Methods:
        plot_monthly_means(data, save=False):
            Generates and optionally saves bar plots of temperature, wind speed, and humidity by month.
        plot_yearly_means(data, save=False):
            Generates and optionally saves bar plots of temperature, wind speed, and humidity by year.
    """

    def __init__(self):
        os.makedirs("plots", exist_ok=True)

    def plot_monthly_means(self, data, save=False):
        mean_temp_per_month = data.groupby('month')['localairtemperature'].mean()
        mean_wind_per_month = data.groupby('month')['localwindspeed'].mean()
        mean_humi_per_month = data.groupby('month')['localhumidity'].mean()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax = ax.ravel()

        mean_temp_per_month.plot(kind='bar', color='skyblue', ax=ax[0])
        mean_wind_per_month.plot(kind='bar', color='skyblue', ax=ax[1])
        mean_humi_per_month.plot(kind='bar', color='skyblue', ax=ax[2])

        ax[0].set_title('Mean Temperature per Month')
        ax[1].set_title('Mean Wind Speed per Month')
        ax[2].set_title('Mean Humidity per Month')
        ax[0].set_xlabel('Month')
        ax[1].set_xlabel('Month')
        ax[2].set_xlabel('Month')
        ax[0].set_ylabel('Mean Temperature (°C)')
        ax[1].set_ylabel('Mean Wind Speed (km/h)')
        ax[2].set_ylabel('Mean Humidity (%)')

        plt.tight_layout()
        if save:
            plt.savefig("plots/monthly_weather_means.png")
        plt.show()

    def plot_yearly_means(self, data, save=False):
        mean_temp_per_year = data.groupby('year')['localairtemperature'].mean()
        mean_wind_per_year = data.groupby('year')['localwindspeed'].mean()
        mean_humi_per_year = data.groupby('year')['localhumidity'].mean()

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax = ax.ravel()

        mean_temp_per_year.plot(kind='bar', color='red', ax=ax[0])
        mean_wind_per_year.plot(kind='bar', color='red', ax=ax[1])
        mean_humi_per_year.plot(kind='bar', color='red', ax=ax[2])

        ax[0].set_title('Mean Temperature per Year')
        ax[1].set_title('Mean Wind Speed per Year')
        ax[2].set_title('Mean Humidity per Year')
        ax[0].set_xlabel('Year')
        ax[1].set_xlabel('Year')
        ax[2].set_xlabel('Year')
        ax[0].set_ylabel('Mean Temperature (°C)')
        ax[1].set_ylabel('Mean Wind Speed (km/h)')
        ax[2].set_ylabel('Mean Humidity (%)')

        plt.tight_layout()
        if save:
            plt.savefig("plots/yearly_weather_means.png")
        plt.show()


class SeeingVisualizer:
    """
    A class for creating visualizations of seeing measurements.

    Methods:
        plot_monthly_seeing(data, save=False):
            Plots monthly average seeing and deviation from minimum.
        plot_minute_seeing_variation(data, save=False):
            Plots per-minute variation in seeing across the day.
    """

    def __init__(self):
        os.makedirs("plots", exist_ok=True)

    def plot_monthly_seeing(self, data, save=False):
        mean_seeing_per_month = data.groupby('month')['Seeing'].mean()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        ax = ax.ravel()

        mean_seeing_per_month.plot(kind='bar', color='red', ax=ax[0])
        (mean_seeing_per_month - min(mean_seeing_per_month)).plot(kind='bar', color='orange', ax=ax[1])

        ax[0].set_title('Mean Seeing per Month')
        ax[1].set_title('Mean Seeing Deviation from Minimum')
        ax[0].set_xlabel('Month')
        ax[1].set_xlabel('Month')
        ax[0].set_ylabel('Mean Seeing (arcsec)')

        plt.tight_layout()
        if save:
            plt.savefig("plots/monthly_seeing.png")
        plt.show()

    def plot_minute_seeing_variation(self, data, save=False):
        minute_seeing_difference = data.groupby([data['hour'], data['minute']])['Seeing'].apply(lambda x: x.max() - x.min())

        plt.figure(figsize=(10, 6))
        minute_seeing_difference.plot(marker='o', linestyle='-')
        plt.title('Difference between Maximum and Minimum Seeing per Minute')
        plt.xlabel('Time (Hour:Minute)')
        plt.ylabel('Difference (Seeing)')
        plt.grid(True)
        plt.xticks(rotation=45)
        if save:
            plt.savefig("plots/minute_seeing_variation.png")
        plt.show()
