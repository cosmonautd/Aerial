import warnings
warnings.filterwarnings(action='ignore')

import numpy
import keras
import tensorflow
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

X = numpy.genfromtxt('haralick-X.csv', delimiter=',')
Y = numpy.genfromtxt('haralick-Y.csv', delimiter=',')

d = X.shape[1]

# Definição fixa das sementes dos geradores aleatórios
# para facilitar a reprodução dos resultados
numpy.random.seed(1)
tensorflow.set_random_seed(1)

# Instanciação do objeto responsável pela divisão de conjuntos de
# treino e teste de acordo com a metodologia K-Fold com K = 5
cross_val = StratifiedKFold(5)
cross_val.get_n_splits(X)

# Total de amostras
total = len(X)
# Variável para contagem da taxa de sucesso
success = 0.0

# Percorre as divisões de conjuntos de treino e teste 
# 5-Fold
for train_index, test_index in cross_val.split(X,numpy.round(Y)):

    # Assinala os conjuntos de treino e teste de acordo
    # com os índices definidos
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    model = keras.models.Sequential([
            keras.layers.Dense(50, activation='sigmoid', input_shape=(d,)),
            keras.layers.Dense(75, activation='sigmoid'),
            keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compilação do modelo
    # Definição do algoritmo de otimização e da função de perda
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

    # Treinamento
    # Executa o algoritmo de otimização, ajustando os pesos das conexões
    # da rede neural com base nos valores de entrada X e saída Y, usando
    # a função de perda como forma de verificar o quão corretas são suas
    # predições durante o treinamento. Realiza 10 passagens pelo conjunto
    # de treinamento. Utiliza 20% dos conjuntos X e Y como validação.
    history = model.fit(X_train, Y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

    # Variáveis para cálculo dos coeficientes de determinação R2 e R2aj
    y = model.predict(X_test).flatten()
    Y_mean = numpy.mean(Y_test)
    SQe = numpy.sum((Y_test - y)**2)
    Syy = numpy.sum((Y_test - Y_mean)**2)

    # Coeficiente de determinação R2
    R2 = 1 - SQe/Syy

    print('R2: %.6f' % (R2))