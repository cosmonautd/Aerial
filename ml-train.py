import warnings
warnings.filterwarnings(action='ignore')

import cv2
import numpy
import keras
import tensorflow
import matplotlib.pyplot as plt

X = numpy.genfromtxt('haralick-X.csv', delimiter=',')
Y = numpy.genfromtxt('haralick-Y.csv', delimiter=',')

d = X.shape[1]

# Definição fixa das sementes dos geradores aleatórios
# para facilitar a reprodução dos resultados
# numpy.random.seed(1)
# tensorflow.set_random_seed(1)

# Instanciação de um modelo sequencial;
# Este modelo é uma pilha de camadas de neurônios;
# Sua construção é feita através da adição sequencial de camadas,
# primeiramente a camada de entrada, depois as camadas e ocultas e, 
# enfim, a camada de saída;
# Neste exemplo, a classe Dense representa camadas totalmente conectadas
model = keras.models.Sequential([
        keras.layers.Dense(50, activation='sigmoid', input_shape=(d,)),
        keras.layers.Dense(75, activation='sigmoid'),
        keras.layers.Dense(1, activation='sigmoid')
])

# Compilação do modelo
# Definição do algoritmo de otimização e da função de perda
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)

# Treinamento
# Executa o algoritmo de otimização, ajustando os pesos das conexões
# da rede neural com base nos valores de entrada X e saída Y, usando
# a função de perda como forma de verificar o quão corretas são suas
# predições durante o treinamento. Realiza 10 passagens pelo conjunto
# de treinamento. Utiliza 20% dos conjuntos X e Y como validação.
history = model.fit(X, Y, epochs=100, validation_split=0.2, callbacks=[early_stopping])

# Visualização da evolução da perda sobre os conjuntos de 
# treinamento e validação
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perda do modelo de cálculo de atravessabilidade')
plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'], loc='upper right')
plt.show()

# Salva a arquitetura da rede em um arquivo JSON
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# Salva os pesos da rede em um arquivo HDF5
model.save_weights("model.h5")