# Seeing_forecast

Predicting astronomical seeing conditions using machine learning to optimize observation schedules.

Predicting atmospheric seeing and humidity conditions at the Observatorio del Roque de los Muchachos (ORM) using historical meteorological and DIMM data.

📌 Overview

This repository contains machine learning and time series models to forecast astronomical seeing and humidity at ORM, leveraging historical data from the Isaac Newton Telescope (INT) and William Herschel Telescope (WHT).
Key Features:

    Data preprocessing for raw meteorological data (INT/WHT).

    Machine learning models for seeing and humidity prediction.

    Time series analysis for seeing forecasts.

🏗️ Project Architecture

Agent Note: This section must be updated every time a new directory or core file is added.

        /data: Contains .csv files for raw and processed atmospheric data.

        /notebooks: Iterative research, EDA, and model prototyping.

        /scripts: Modular Python logic for cleaning, training, and inference.

        /models: Serialized model files (.pkl, .joblib, or .h5).

        /skills: Standard Operating Procedures (SOPs) for the AI Agent.

        /logs: Active session memory and error tracking.

🛠️ Usage

1. Jupyter Notebooks

    ML_humidity.ipynb: Trains ML models to predict humidity from meteorological data.

    ML_seeing.ipynb: Trains ML models to predict seeing from meteorological data.

    time_series.ipynb: Uses time series methods (e.g., AR(p)) for seeing forecasting (UNDER DEVELOPMENT).

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


🤖 AI Orchestration (AGENT.md)

This repository is designed to be managed by an AI Agent (Cursor/LLM).

    - Operational Protocol: The agent follows AGENT.md for session management and coding standards.

    - Skill-Based Execution: Technical implementation details for preprocessing, modeling, and plotting are governed by the /skills directory.

    - Documentation Sync Rule: The agent is strictly required to update this README whenever the repository structure changes.

📈 Modeling Protocol

Following scientific best practices, every model in this project undergoes:

    1. Imbalance Analysis: Mandatory check for class distribution.

    2. Triple-Split: Data is partitioned into Train, Validation, and Test sets.

    3. Uncertainty Quantification: All results are plotted showing Standard Deviation and IQR (25%/75%).

    4. Optimization: Models are selected based on Best Fit (calibration) first, then Performance metrics.

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
