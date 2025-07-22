# Seeing_forecast

Seeing Forecast for Astronomical Observations

Predicting atmospheric seeing and humidity conditions at the Observatorio del Roque de los Muchachos (ORM) using historical meteorological and DIMM data.
📌 Overview

This repository contains machine learning and time series models to forecast astronomical seeing and humidity at ORM, leveraging historical data from the Isaac Newton Telescope (INT) and William Herschel Telescope (WHT).
Key Features:

    Data preprocessing for raw meteorological data (INT/WHT).

    Machine learning models for seeing and humidity prediction.

    Time series analysis for seeing forecasts.

📂 Repository Structure

Seeing_forecast/
└── main.py                       # Main script for processing, cleaning, visualization and ML modelling of the data
├── data/  
│   ├── INT_data/                 # Historical meteorological data (INT)  
│   ├── WHT_data/                 # Historical meteorological data (WHT)  
│   └── seeing_data.csv           # DIMM seeing measurements from WHT  
│  
├── notebooks/  
│   ├── ML_humidity.ipynb         # Humidity prediction (preprocessing + ML)  
│   ├── ML_seeing.ipynb           # Seeing prediction (preprocessing + ML)  
│   └── time_series.ipynb         # Time series forecasting for seeing  
│  
└── src/  
|    └── pre_process.py            # Script to preprocess raw INT/WHT data
|    ├── ml_modelling.py           # Script for ML models selection (DNN, RFR, Polynomial) and training
|    ├── visualizer.py             # Visualization and plots of weather and seeing data
|    └── preprocess/
|          ├── int_cleaner.py      # Loading and preprocessing of INT weather data
|          ├── wht_cleaner.py      # Loading and preprocessing of WHT weather data
|          ├── seeing_cleaner.py   # Loading and preprocessing archival seeing data
|          └── utils/
|                └── resampler.py       # Script to weather data to a specified time frequency
|
└── plots/                         # Folder to save the plots

🛠️ Usage

1. Jupyter Notebooks

    ML_humidity.ipynb: Trains ML models to predict humidity from meteorological data.

    ML_seeing.ipynb: Trains ML models to predict seeing from meteorological data.

    time_series.ipynb: Uses time series methods (e.g., AR(p)) for seeing forecasting.

Open notebooks in Jupyter Lab or Google Colab:
bash

jupyter-lab notebooks/ML_seeing.ipynb

📊 Data Sources

    Meteorological Data: Historical records from INT/WHT.

    Seeing Data: DIMM measurements from WHT (seeing_data.csv).

🤖 Models

    Machine Learning:

        Regression (Random Forest, Polynomial regression, and DNN) for seeing/humidity prediction.

    Time Series:

        AR(p) for seeing and humidity forecasting. Further implementation of SARIMA model is coming.

📝 Requirements

You will need to install the dependencies (in a dedicated environment) for running the scripts. It is highly recommended to use Conda:

For Linux:
```
conda env create -f environment_linux.yml
```

then, you have to activate the environment:
```
conda activate ml_env_linux
```

For Windows:
```
conda env create -f environment_win.yml
```
to activate the environment using conda:
```
acc_win
```
