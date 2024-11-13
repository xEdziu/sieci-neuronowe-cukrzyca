from data_preparation import load_and_prepare_data
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

# Wczytanie i przygotowanie danych
print('Wczytywanie danych... -', os.path.basename(__file__))
X_train, X_test, y_train, y_test = load_and_prepare_data(data_path='data/diabetes.csv')

# Wczytanie wytrenowanego modelu
print('Wczytywanie modelu... -', os.path.basename(__file__))
model = load_model('models/model-v1.keras')

# Przewidywania na zbiorach treningowym i walidacyjnym
print('Przewidywanie... -', os.path.basename(__file__))
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Używamy progu 0.5 do klasyfikacji 0-1
print('Klasyfikacja... -', os.path.basename(__file__))
train_predictions = (train_predictions > 0.5).astype(int)
test_predictions = (test_predictions > 0.5).astype(int)

# Obliczenie MS
print('Obliczanie MSE... -', os.path.basename(__file__))
train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f'MSE dla zbioru treningowego: {train_mse}')
print(f'MSE dla zbioru walidacyjnego: {test_mse}')

# Wykres MSE dla treningu i walidacji
plt.plot([0], [train_mse], label='MSE treningowy', marker='o')  # MSE po treningu
plt.plot([0], [test_mse], label='MSE walidacyjny', marker='o')   # MSE po walidacji
plt.xlabel('Epoki (model wytrenowany)')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.title('Wykres MSE na zapisanym modelu')
plt.show()
