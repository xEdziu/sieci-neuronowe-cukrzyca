import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_prepare_data(data_path='data/diabetes.csv'):
    """
    Wczytuje dane, koduje kategorie, dzieli na zbiór treningowy i testowy oraz normalizuje.
    :param data_path: Ścieżka do pliku CSV z danymi.
    :return: X_train, X_test, y_train, y_test
    """
    # Wczytanie danych
    print('Wczytywanie danych... -', os.path.basename(__file__))
    data = pd.read_csv(data_path)
    
    # Oddzielenie cech (X) i etykiety (y)
    print('Przygotowywanie danych... -', os.path.basename(__file__))
    X = data.drop(columns=['diabetes'])
    y = data['diabetes']
    
    # Kodowanie kolumn kategorycznych za pomocą kodowania one-hot
    print('Kodowanie kolumn kategorycznych... -', os.path.basename(__file__))
    X = pd.get_dummies(X, columns=['gender', 'smoking_history'], drop_first=True)
    
    # Podział na zbiór treningowy i testowy (80%-20%)
    print('Dzielenie danych na zbiór treningowy i testowy... -', os.path.basename(__file__))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizacja cech przy użyciu StandardScaler
    print('Normalizacja danych... -', os.path.basename(__file__))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print('Gotowe - dane przygotowane! -', os.path.basename(__file__))
    return X_train, X_test, y_train, y_test
