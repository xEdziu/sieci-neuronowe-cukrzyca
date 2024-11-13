from data_preparation import load_and_prepare_data
from model import create_model
import matplotlib.pyplot as plt
import os

# Wczytanie i przygotowanie danych
X_train, X_test, y_train, y_test = load_and_prepare_data(data_path='data/diabetes.csv')

# Utworzenie modelu
model = create_model(input_shape=X_train.shape[1])

# Trenowanie modelu
print('Trenowanie modelu... -', os.path.basename(__file__))
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Wizualizacja strat (binary crossentropy) na danych treningowych i walidacyjnych
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.xlabel('Epoki')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.title('Wykres błędu Binary Crossentropy')
plt.show()

# Wizualizacja dokładności na danych treningowych i walidacyjnych
plt.plot(history.history['accuracy'], label='Dokładność treningowa')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()
plt.title('Wykres dokładności')
plt.show()

choice = input('Czy zapisać model? (t/n): ')
if choice.lower() == 't':
    name = input('Podaj nazwę modelu: ')
    model.save(f'models/{name}.keras')
    print('Model zapisany!')
else:
    print('Model nie został zapisany.')
    
print('Koniec skryptu -', os.path.basename(__file__))