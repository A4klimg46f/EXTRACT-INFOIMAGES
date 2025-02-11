# EXTRACT-INFOIMAGES
## Código de ML para extração de informações de imagens de satélite

# Processamento e Classificação de Imagens Raster com Random Forest e Redes Neurais Convolucionais (CNNs)

Este projeto demonstra o uso de ferramentas de aprendizado de máquina e deep learning para processamento e classificação de imagens raster. Utilizando bibliotecas como rasterio, numpy, 
scikit-learn e tensorflow, o código abrange desde a manipulação básica de imagens até a criação de modelos avançados de classificação.

## Requisitos
Para executar este código, as seguintes bibliotecas Python devem estar instaladas:
  - rasterio
  - numpy
  - matplotlib
  - scikit-learn
  - tensorflow

Você pode instalar essas dependências com o comando:
pip install rasterio numpy matplotlib scikit-learn tensorflow

## Estrutura do Código
1. Configuração do Ambiente
   - O ambiente é configurado para manipular imagens raster e implementar modelos de aprendizado de máquina. Certifique-se de ter o caminho correto para a imagem raster (imagem.tif).
2. Leitura e Pré-processamento
     - As imagens raster são carregadas utilizando rasterio, e as bandas são normalizadas para garantir a uniformidade dos dados.
Além disso, um índice NDVI (Normalized Difference Vegetation Index) é calculado com base nas bandas NIR e RED.
3. Modelo Random Forest
     - Um modelo de Random Forest é usado para classificar os pixels das imagens com base em dados normalizados.
A entrada é dividida em dados de treinamento e teste, e o modelo é avaliado utilizando a métrica de acurácia.
4. Rede Neural Convolucional (CNN)
     - Uma CNN simples é construída com TensorFlow/Keras para classificação de imagens. Os dados de entrada são ajustados para a estrutura esperada por uma CNN,
e o modelo é treinado e validado com base nos dados fornecidos.
5. Exportação de Resultados
     - Os resultados do modelo Random Forest são exportados como um arquivo raster (resultado_classificacao.tif), com a mesma referência espacial da imagem de entrada.
6. Visualização
     - O resultado da classificação é visualizado como um mapa utilizando matplotlib.
## Instruções de Uso
1. Atualize o Caminho da Imagem: Certifique-se de especificar o caminho correto para a imagem raster na variável image_path.
2. Executar o Código: Execute o script em um ambiente Python configurado com as dependências necessárias.
3. Interpretação dos Resultados:
     - Acurácia do modelo Random Forest é exibida no console.
     - O modelo CNN é treinado e validado, exibindo as métricas ao final.
     - O mapa classificado é salvo em um arquivo GeoTIFF e também exibido em uma janela gráfica.
## Notas Importantes
      - Os dados de exemplo para o modelo Random Forest (rótulos) são gerados de forma aleatória e devem ser substituídos por dados reais para aplicações práticas.
      - O reshape para a CNN (128x128x3) deve ser ajustado de acordo com o tamanho e o formato dos seus dados reais.
      - Este é um exemplo educacional e pode exigir ajustes para ser usado em um cenário de produção.
      
