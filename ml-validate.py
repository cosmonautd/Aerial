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

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

def baseline_model():
    model = keras.models.Sequential([
            keras.layers.Dense(50, activation='sigmoid', input_shape=(d,)),
            keras.layers.Dense(100, activation='sigmoid'),
            keras.layers.Dense(50, activation='sigmoid'),
            keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

kfold = StratifiedKFold(n_splits=10, random_state=1)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))