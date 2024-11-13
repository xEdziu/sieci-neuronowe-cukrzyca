from data_preparation import load_and_prepare_data
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
import os

# Wczytanie i przygotowanie danych
print('Wczytywanie danych... -', os.path.basename(__file__))
X_train, X_test, y_train, y_test = load_and_prepare_data(data_path='data/diabetes.csv')

# Wczytanie wytrenowanego modelu
print('Wczytywanie modelu... -', os.path.basename(__file__))
model = load_model('models/model-v1.keras')  # Zmień nazwę modelu, jeśli jest inna

# Pobranie wag z warstw modelu
print('Pobieranie wag... -', os.path.basename(__file__))
hidden_layer_weights = model.layers[0].get_weights()[0]  # Wagi warstwy ukrytej
hidden_layer_biases = model.layers[0].get_weights()[1]   # Przesunięcia warstwy ukrytej

output_weights = model.layers[1].get_weights()[0]  # Wagi warstwy wyjściowej
output_biases = model.layers[1].get_weights()[1]   # Przesunięcia warstwy wyjściowej

# Wizualizacja wag dla warstwy ukrytej z etykietami
plt.figure(figsize=(10, 5))
plt.bar(range(len(hidden_layer_weights.flatten())), hidden_layer_weights.flatten(), color='blue')
plt.xlabel('Indeks wagi')
plt.ylabel('Wartość wagi')
plt.title('Wizualizacja wag w warstwie ukrytej')
plt.show()

# Wizualizacja przesunięć (bias) dla warstwy ukrytej z etykietami
plt.figure(figsize=(10, 5))
plt.bar(range(len(hidden_layer_biases)), hidden_layer_biases, color='orange')
for i, value in enumerate(hidden_layer_biases):
    plt.text(i, value, f'{value:.2f}', ha='center', va='bottom' if value > 0 else 'top', fontsize=8)
plt.xlabel('Indeks przesunięcia')
plt.ylabel('Wartość przesunięcia')
plt.title('Wizualizacja przesunięć (bias) w warstwie ukrytej')
plt.show()

# Pobranie wag i przesunięć z warstwy wyjściowej (drugiej warstwy Dense)
output_layer_weights = model.layers[1].get_weights()[0]  # Wagi warstwy wyjściowej
output_layer_biases = model.layers[1].get_weights()[1]   # Przesunięcia warstwy wyjściowej

# Wizualizacja wag dla warstwy wyjściowej z etykietami
plt.figure(figsize=(10, 5))
plt.bar(range(len(output_layer_weights.flatten())), output_layer_weights.flatten(), color='green')
for i, value in enumerate(output_layer_weights.flatten()):
    plt.text(i, value, f'{value:.2f}', ha='center', va='bottom' if value > 0 else 'top', fontsize=8)
plt.xlabel('Indeks wagi')
plt.ylabel('Wartość wagi')
plt.title('Wizualizacja wag w warstwie wyjściowej')
plt.show()

# Wizualizacja przesunięcia (bias) dla warstwy wyjściowej
plt.figure(figsize=(5, 5))
plt.plot(output_layer_biases, color='red', marker='o')
for i, value in enumerate(output_layer_biases):
    plt.text(i, value, f'{value:.2f}', ha='center', va='bottom' if value > 0 else 'top', fontsize=8)
plt.xlabel('Indeks przesunięcia')
plt.ylabel('Wartość przesunięcia')
plt.title('Wizualizacja przesunięcia (bias) w warstwie wyjściowej')
plt.show()

# Przewidywania na zbiorach treningowym i walidacyjnym
print('Przewidywanie... -', os.path.basename(__file__))
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Używamy progu 0.5 do klasyfikacji 0-1
print('Klasyfikacja... -', os.path.basename(__file__))
train_predictions_binary = (train_predictions > 0.5).astype(int)
test_predictions_binary = (test_predictions > 0.5).astype(int)

# Obliczenie MSE
print('Obliczanie MSE... -', os.path.basename(__file__))
train_mse = mean_squared_error(y_train, train_predictions_binary)
test_mse = mean_squared_error(y_test, test_predictions_binary)

print(f'MSE dla zbioru treningowego: {train_mse}')
print(f'MSE dla zbioru walidacyjnego: {test_mse}')

# Wykres MSE dla treningu i walidacji
plt.plot([0], [train_mse], label='MSE treningowy', marker='o')  # MSE po treningu
plt.plot([0], [test_mse], label='MSE walidacyjny', marker='o')   # MSE po walidacji
plt.xlabel('Epoki (model wytrenowany)')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.title('Wykres MSE na zapisanym modelu v1')
plt.show()

# Obliczenie błędów klasyfikacji
print('Obliczanie błędów klasyfikacji... -', os.path.basename(__file__))
train_error = 1 - accuracy_score(y_train, train_predictions_binary)
test_error = 1 - accuracy_score(y_test, test_predictions_binary)

print(f'Błąd klasyfikacji dla zbioru treningowego: {train_error:.4f}')
print(f'Błąd klasyfikacji dla zbioru walidacyjnego: {test_error:.4f}')

# Wykres błędu klasyfikacji
plt.bar_label(plt.bar([f'Błąd treningowy', f'Błąd walidacyjny'], [train_error, test_error], color=['blue', 'orange']))
plt.xlabel('Zbiór danych')
plt.ylabel('Błąd klasyfikacji')
plt.title('Wykres błędu klasyfikacji na zapisanym modelu v1')
plt.show()