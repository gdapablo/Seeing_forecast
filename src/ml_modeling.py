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
    Clase para preparar y combinar los datos necesarios para el modelado.

    Métodos:
    --------
    prepare_data(data_resample: pd.DataFrame, seeing: pd.DataFrame) -> pd.DataFrame
        Combina y limpia los datos resampleados y de seeing para análisis de ML.
    """

    @staticmethod
    def prepare_data(data_resample: pd.DataFrame, seeing: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara los datos para modelado.

        Inputs:
        -------
        data_resample : pd.DataFrame
            Datos meteorológicos resampleados.
        seeing : pd.DataFrame
            Datos de seeing con columnas año, mes, día, hora, minuto, segundo.

        Output:
        -------
        combined : pd.DataFrame
            DataFrame combinado y limpio listo para ML.
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
    Clase para preprocesar los datos combinados y preparar variables para ML.

    Métodos:
    --------
    preprocess(combined: pd.DataFrame) -> tuple
        Devuelve los datos separados en train/test listos para modelar y el scaler de features.
    """

    def preprocess(self, combined: pd.DataFrame) -> tuple:
        """
        Preprocesa datos: convierte dummies, escala variables, separa features y target.

        Input:
        ------
        combined : pd.DataFrame
            DataFrame combinado de datos resampleados y seeing.

        Output:
        -------
        (X_train, X_test, y_train, y_test) : tuple
            Datos de entrenamiento y test separados.
        scaler : sklearn.preprocessing.MinMaxScaler
            Scaler usado para escalar columnas numéricas.
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

        cols_to_scale = ['windspeed', 'airtemperature', 'relativehumidity']
        scaler = preprocessing.MinMaxScaler()
        data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

        return train_test_split(data, target, test_size=0.2, random_state=101), scaler


class RandomForestModel:
    """
    Modelo de regresión basado en Random Forest.

    Métodos:
    --------
    train_and_evaluate(X_train, X_test, y_train, y_test) -> RandomForestRegressor
        Entrena y evalúa el modelo, imprime métricas.
    """

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Entrena el modelo y calcula MAE, MSE y R2 sobre test.

        Inputs:
        -------
        X_train, X_test : pd.DataFrame o np.array
            Datos de entrada para entrenamiento y prueba.
        y_train, y_test : pd.Series o np.array
            Valores objetivo para entrenamiento y prueba.

        Output:
        -------
        model : RandomForestRegressor
            Modelo entrenado.
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
    Modelo de regresión polinómica para diferentes grados.

    Métodos:
    --------
    evaluate_polynomial_degrees(X_train, y_train, X_test, y_test, max_degree=6) -> None
        Evalúa y grafica métricas para diferentes grados del polinomio.
    """

    def evaluate_polynomial_degrees(self, X_train, y_train, X_test, y_test, max_degree=6):
        """
        Entrena modelos polinómicos y grafica MAE, MSE y R2 en train para grados 1 a max_degree.

        Inputs:
        -------
        X_train, X_test, y_train, y_test : datasets para entrenamiento y prueba
        max_degree : int
            Grado máximo del polinomio a evaluar.

        Output:
        -------
        None (genera gráficos)
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
    Modelo de red neuronal densa para regresión.

    Métodos:
    --------
    train_and_evaluate(X_train, X_test, y_train, y_test) -> None
        Entrena, evalúa el modelo y genera gráficos de diagnóstico.
    """

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
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

        model.fit(X_train, y_train_scaled, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

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
        plt.show()

        df_comparison = pd.DataFrame({'Real': y_test_scaled.flatten(), 'Predicted': y_pred.flatten()})

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_comparison)
        plt.title('Boxplot of Real vs Predicted (DNN)')
        plt.show()

        residuals = y_test_scaled.flatten() - y_pred.flatten()
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, edgecolor='k')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Histogram of Residuals (DNN)')
        plt.show()


class MLModelingPipeline:
    """
    Pipeline para ejecutar el flujo completo de ML: preparación, preprocesamiento y entrenamiento/evaluación de modelos.

    Métodos:
    --------
    run_model(data_resample, seeing, model_name) -> None
        Ejecuta el modelo especificado ('random_forest', 'polynomial', 'dnn').
    """

    def __init__(self):
        self.preparer = DataPreparer()
        self.preprocessor = DataPreprocessor()
        self.rf_model = RandomForestModel()
        self.poly_model = PolynomialRegressionModel()
        self.dnn_model = DenseNeuralNetworkModel()

    def run_model(self, data_resample: pd.DataFrame, seeing: pd.DataFrame, model_name: str) -> None:
        """
        Corre el modelo indicado por model_name con los datos dados.

        Inputs:
        -------
        data_resample : pd.DataFrame
            Datos meteorológicos resampleados.
        seeing : pd.DataFrame
            Datos de seeing.
        model_name : str
            Nombre del modelo a ejecutar. Opciones: 'random_forest', 'polynomial', 'dnn'.

        Output:
        -------
        None (imprime resultados y muestra gráficos según modelo)
        """
        combined = self.preparer.prepare_data(data_resample, seeing)
        (X_train, X_test, y_train, y_test), _ = self.preprocessor.preprocess(combined)

        if model_name == 'random_forest':
            self.rf_model.train_and_evaluate(X_train, X_test, y_train, y_test)
        elif model_name == 'polynomial':
            self.poly_model.evaluate_polynomial_degrees(X_train, y_train, X_test, y_test)
        elif model_name == 'dnn':
            self.dnn_model.train_and_evaluate(X_train, X_test, y_train, y_test)
        else:
            print(f"Modelo desconocido: {model_name}. Opciones válidas: 'random_forest', 'polynomial', 'dnn'.")
