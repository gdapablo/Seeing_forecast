import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class DataPreparer:
    """
    Class for preleminarly prepare and combine the seeing and met data for further training.

    Methods:
    --------
    prepare_data(data_resample: pd.DataFrame, seeing: pd.DataFrame) -> pd.DataFrame
        Combine and clean seeing data for ML train preparation.
    """

    @staticmethod
    def prepare_data(data_resample: pd.DataFrame, seeing: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the data for modelling.

        Inputs:
        -------
        data_resample : pd.DataFrame
            Resampled weather data.
        seeing : pd.DataFrame
            Seeing data with columns year, month, day, hour, minute, second.

        Output:
        -------
        combined : pd.DataFrame
            Combined DataFrame ready for ML training.
        """
        seeing['Datetime'] = pd.to_datetime(seeing[['year', 'month', 'day', 'hour', 'minute', 'second']])
        seeing_resample = seeing.resample('10T', on='Datetime').mean()
        seeing_resample.reset_index(inplace=True)
        seeing_resample['year'] = seeing_resample.Datetime.dt.year
        seeing_resample['month'] = seeing_resample.Datetime.dt.month
        seeing_resample['day'] = seeing_resample.Datetime.dt.day
        seeing_resample['hour'] = seeing_resample.Datetime.dt.hour
        seeing_resample['minute'] = seeing_resample.Datetime.dt.minute
        seeing_resample.drop('Datetime', axis=1, inplace=True)

        combined = pd.merge(data_resample, seeing_resample,
                            on=['year', 'month', 'day', 'hour', 'minute'], how='right')
        combined.dropna(inplace=True)
        return combined


class DataPreprocessor:
    """
    Class for preprocessing data and combine them for ML training.

    Methods:
    --------
    preprocess(combined: pd.DataFrame) -> tuple
        Returns the data as train and test datasets ready for modeling and for scale the features.
    """

    def preprocess(self, combined: pd.DataFrame) -> tuple:
        """
        Preprocessing data: convert into dummy variables, scale the variables, split between features and target.

        Input:
        ------
        combined : pd.DataFrame
            Combined DataFrame of resampled data and seeing.

        Output:
        -------
        (X_train, X_test, y_train, y_test) : tuple
            Training and testing sets.
        scaler : sklearn.preprocessing.MinMaxScaler
            Scaler used for scaling the numerical columns.
        """
        df = combined.copy()
        if 'seeing_diff' in df.columns:
            df.drop(columns=['seeing_diff'], inplace=True)

        df.reset_index(drop=True, inplace=True)
        target = df['Seeing']
        data = df.drop(columns='Seeing')

        cols_to_int = ['hour', 'day', 'month', 'year', 'minute']
        data = pd.get_dummies(data, columns=cols_to_int)

        bool_cols = data.select_dtypes(include=['bool']).columns
        data[bool_cols] = data[bool_cols].astype('int32')

        cols_to_scale = ['localwindspeed', 'localairtemperature', 'localhumidity']
        scaler = preprocessing.MinMaxScaler()
        data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

        return train_test_split(data, target, test_size=0.2, random_state=101), scaler


class RandomForestModel:
    """
    Random Forest model regression.

    Methods:
    --------
    train_and_evaluate(X_train, X_test, y_train, y_test) -> RandomForestRegressor
        Train and evaluate the model show the metrics as output.
    """

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, save=False):
        """
        Training the model and evaluates MAE, MSE and R2 over test.

        Inputs:
        -------
        X_train, X_test : pd.DataFrame o np.array
            input data for train and testing sets.
        y_train, y_test : pd.Series o np.array
            target values for training and testing.

        Output:
        -------
        model : RandomForestRegressor
            Trained model.
        """
        model = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Random Forest Regressor:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  R2: {r2:.4f}\n")

        return model


class PolynomialRegressionModel:
    """
    Polynomial regresion model for different degrees.

    Methods:
    --------
    evaluate_polynomial_degrees(X_train, y_train, X_test, y_test, max_degree=6) -> None
        Evaluate and plot the metrics for different polynomial degrees.
    """

    def evaluate_polynomial_degrees(self, X_train, y_train, X_test, y_test, max_degree=6, save=False):
        """
        Train and plot polynomial models shwoing MAE, MSE y R2 from degree 1 to max_degree.

        Inputs:
        -------
        X_train, X_test, y_train, y_test : training and testing datasets
        max_degree : int
            max degree of the polynomial

        Output:
        -------
        None (generate plots)
        """
        mse_train_list = []
        mae_train_list = []
        r2_train_list = []

        for degree in range(1, max_degree + 1):
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)

            model = LinearRegression()
            model.fit(X_train_poly, y_train)

            y_train_pred = model.predict(X_train_poly)

            mae_train = mean_absolute_error(y_train, y_train_pred)
            mse_train = mean_squared_error(y_train, y_train_pred)
            r2_train = r2_score(y_train, y_train_pred)

            mae_train_list.append(mae_train)
            mse_train_list.append(mse_train)
            r2_train_list.append(r2_train)

        degrees = list(range(1, max_degree + 1))

        plt.figure()
        plt.plot(degrees, mse_train_list, marker='o', label='MSE')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('MSE')
        plt.title('Polynomial Regression - Training MSE')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(degrees, mae_train_list, marker='o', label='MAE')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('MAE')
        plt.title('Polynomial Regression - Training MAE')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(degrees, r2_train_list, marker='o', label='R2')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('R2 Score')
        plt.title('Polynomial Regression - Training R2')
        plt.legend()
        plt.show()


class DenseNeuralNetworkModel:
    """
    Regression DNN.

    Methods:
    --------
    train_and_evaluate(X_train, X_test, y_train, y_test) -> None
        Train, evaluate and plot DNN model.
    """

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, save=False):
        """
        Train a DNN model and estimates MSE, MAE, R2 and generate plots

        Inputs:
        -------
        X_train, X_test, y_train, y_test : training and testing datasets.

        Output:
        -------
        None (Print metrics and generate plots)
        """
        scaler_y = preprocessing.MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

        model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        model.fit(X_train, y_train_scaled, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

        loss, mae = model.evaluate(X_test, y_test_scaled, verbose=0)
        print("Dense Neural Network:")
        print(f"  Mean Squared Error (MSE): {loss:.4f}")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test_scaled, y_pred)
        print(f"  R2: {r2:.4f}")

        plt.figure(figsize=(8, 8))
        plt.scatter(y_test_scaled, y_pred, alpha=0.5)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('Real values (scaled)')
        plt.ylabel('Predicted values')
        plt.title('DNN Predictions vs Real Values')
        if save:
            plt.savefig("plots/dnn_predicted_vs_real.png")
        plt.show()

        df_comparison = pd.DataFrame({'Real': y_test_scaled.flatten(), 'Predicted': y_pred.flatten()})

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_comparison)
        plt.title('Boxplot of Real vs Predicted (DNN)')
        if save:
            plt.savefig("plots/dnn_boxplot.png")
        plt.show()

        residuals = y_test_scaled.flatten() - y_pred.flatten()
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, edgecolor='k')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Histogram of Residuals (DNN)')
        if save:
            plt.savefig("plots/dnn_histogram.png")
        plt.show()


class MLModelingPipeline:
    """
    Pipeline to execute ML models: preliminars, preprocessing and training/evaluation.

    Methods:
    --------
    run_model(data_resample, seeing, model_name) -> None
        Execute the specified model ('random_forest', 'polynomial', 'dnn').
    """

    def __init__(self):
        self.preparer = DataPreparer()
        self.preprocessor = DataPreprocessor()
        self.rf_model = RandomForestModel()
        self.poly_model = PolynomialRegressionModel()
        self.dnn_model = DenseNeuralNetworkModel()

    def run_model(self, data_resample: pd.DataFrame, seeing: pd.DataFrame, model_name: str, save=False) -> None:
        """
        Runs the model_name with the specified data.

        Inputs:
        -------
        data_resample : pd.DataFrame
            Met weather data resampled.
        seeing : pd.DataFrame
            Seeing data.
        model_name : str
            Name of the model. Options: 'random_forest', 'polynomial', 'dnn'.

        Output:
        -------
        None (print the results and show plots according to the model)
        """
        combined = self.preparer.prepare_data(data_resample, seeing)
        (X_train, X_test, y_train, y_test), _ = self.preprocessor.preprocess(combined)

        if model_name == 'random_forest':
            self.rf_model.train_and_evaluate(X_train, X_test, y_train, y_test, save=False)
        elif model_name == 'polynomial':
            self.poly_model.evaluate_polynomial_degrees(X_train, y_train, X_test, y_test, save=False)
        elif model_name == 'dnn':
            self.dnn_model.train_and_evaluate(X_train, X_test, y_train, y_test, save=False)
        else:
            print(f"Unknown model: {model_name}. Valid options: 'random_forest', 'polynomial', 'dnn'.")
