from data_preparation import load_and_prepare_data
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import codecs
import os

output_file_path = os.path.join("results","v2","model_results.txt")
os.makedirs("results", exist_ok=True)

# Wczytanie i przygotowanie danych
print('Wczytywanie danych... -', os.path.basename(__file__))
X_train, X_test, y_train, y_test = load_and_prepare_data(data_path='data/diabetes.csv')

# Wczytanie wytrenowanego modelu
print('Wczytywanie modelu... -', os.path.basename(__file__))
model = load_model('models/model-v2.keras')  # Zmień nazwę modelu, jeśli jest inna
model.summary() # Wyświetlenie podsumowania modelu

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

# Obliczenie macierzy pomyłek
cm = confusion_matrix(y_test, test_predictions_binary)

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Przewidywane 0', 'Przewidywane 1'], yticklabels=['Rzeczywiste 0', 'Rzeczywiste 1'])
plt.xlabel('Przewidywana klasa')
plt.ylabel('Rzeczywista klasa')
plt.title('Macierz pomyłek')
plt.show()

# Obliczenie metryk
train_accuracy = accuracy_score(y_train, train_predictions_binary)
test_accuracy = accuracy_score(y_test, test_predictions_binary)
train_mse = mean_squared_error(y_train, train_predictions_binary)
test_mse = mean_squared_error(y_test, test_predictions_binary)
cm = confusion_matrix(y_test, test_predictions_binary)

# Obliczenie precyzji, czułości i F1-score na zbiorze walidacyjnym
precision = precision_score(y_test, test_predictions_binary)
recall = recall_score(y_test, test_predictions_binary)
f1 = f1_score(y_test, test_predictions_binary)

print(f'Precyzja: {precision:.4f}')
print(f'Czułość: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

# Zapis wyników do pliku tekstowego
with codecs.open(output_file_path, "w", "utf-8-sig") as file:
    file.write("Wyniki modelu sieci neuronowej\n")
    file.write("====================================\n")
    file.write(f"Dokładność na zbiorze treningowym: {train_accuracy:.4f}\n")
    file.write(f"Dokładność na zbiorze walidacyjnym: {test_accuracy:.4f}\n")
    file.write("\nMacierz pomyłek (Confusion Matrix):\n")
    file.write(f"{cm}\n")
    file.write("====================================\n")
    file.write("Wagi warstwy ukrytej:\n")
    file.write(f"{hidden_layer_weights}\n")
    file.write("Przesunięcia warstwy ukrytej:\n")
    file.write(f"{hidden_layer_biases}\n")
    file.write("Wagi warstwy wyjściowej:\n")
    file.write(f"{output_layer_weights}\n")
    file.write("Przesunięcia warstwy wyjściowej:\n")
    file.write(f"{output_layer_biases}\n")
    file.write("====================================\n")
    file.write("\nDodatkowe metryki klasyfikacji:\n")
    file.write(f"Precyzja: {precision:.4f}\n")
    file.write(f"Czułość: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n")

print(f"Wyniki zostały zapisane w pliku: {output_file_path}")