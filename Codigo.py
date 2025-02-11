import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------ Configuração do Ambiente ------------------
# Ambiente configurado para manipulação de imagens e aprendizado de máquina

# ------------------ Leitura de Imagens ------------------
# Leitura de imagens raster com Rasterio
image_path = 'caminho/para/imagem.tif'
with rasterio.open(image_path) as src:
    image = src.read()  # Lê todas as bandas
    transform = src.transform  # Obtem a informação de transformação geoespacial

# ------------------ Normalização ------------------
def normalize_band(band):
    return (band - np.min(band)) / (np.max(band) - np.min(band))

# Normalizar cada banda
image_normalized = np.array([normalize_band(band) for band in image])

# ------------------ Índice NDVI ------------------
def calculate_ndvi(nir_band, red_band):
    return (nir_band - red_band) / (nir_band + red_band)

ndvi = calculate_ndvi(image_normalized[4], image_normalized[3])  # Exemplo para Sentinel-2

# ------------------ Random Forest ------------------
# Geração de dados de exemplo para treinamento
features = image_normalized.reshape(-1, image_normalized.shape[0])  # Reorganiza dados
labels = np.random.randint(0, 2, size=(features.shape[0],))  # Rótulos aleatórios para exemplo

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Treinamento do Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predição e avaliação
y_pred = rf.predict(X_test)
print("Acurácia Random Forest:", accuracy_score(y_test, y_pred))

# ------------------ Redes Neurais Convolucionais (CNNs) ------------------
# Preparar dados para CNN
X_train_cnn = X_train.reshape(-1, 128, 128, 3)  # Exemplo de reshape para CNN
X_test_cnn = X_test.reshape(-1, 128, 128, 3)

# Construção da CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compilação do modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinamento
model.fit(X_train_cnn, y_train, epochs=10, validation_data=(X_test_cnn, y_test))

# ------------------ Exportação dos Resultados ------------------
# Exemplo de exportação de mapa classificado
classified_image = rf.predict(features).reshape(image.shape[1], image.shape[2])

with rasterio.open(
    'resultado_classificacao.tif',
    'w',
    driver='GTiff',
    height=classified_image.shape[0],
    width=classified_image.shape[1],
    count=1,
    dtype=classified_image.dtype,
    crs=src.crs,
    transform=transform
) as dst:
    dst.write(classified_image, 1)

# ------------------ Visualização ------------------
plt.figure(figsize=(10, 6))
plt.imshow(classified_image, cmap='jet')
plt.title('Mapa Classificado')
plt.colorbar()
plt.show()
