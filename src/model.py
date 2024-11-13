from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import os

def create_model(input_shape):
    """
    Tworzy model sieci neuronowej o strukturze dwuwarstwowej.
    :param input_shape: Rozmiar wejścia (liczba cech).
    :return: model sieci neuronowej
    """
    # Inicjalizacja modelu jako sekwencyjnego
    model = Sequential([
        Input(shape=(input_shape,)),           # Warstwa wejściowa z określonym kształtem
        Dense(10, activation='relu'),          # Warstwa ukryta
        Dense(1, activation='sigmoid')         # Warstwa wyjściowa
    ])
    
    # Kompilacja modelu
    print('Kompilacja modelu...')
    
    #Ustawienie optymalizatora, funkcji straty i metryki
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    print('Gotowe - model utworzony! -', os.path.basename(__file__))
    return model
