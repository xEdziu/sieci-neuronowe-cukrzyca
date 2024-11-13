from keras.models import load_model
from keras_visualizer import visualizer

# Pobranie modeli z plików
model_paths = ['models/model-v1.keras', 'models/model-v2.keras']

# Wizualizacja modeli
for model_path in model_paths:
    model = load_model(model_path)  # Wczytanie modelu z pliku
    filename = model_path.split('/')[-1].split('.')[0]
    filename_path = "results/visualizations/" + filename
    visualizer(model, file_name=filename_path, file_format='pdf', view=True)
    visualizer(model, file_name=filename_path, file_format='png', view=False)
