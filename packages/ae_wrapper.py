import packages
from tensorflow import keras
from sklearn.base import BaseEstimator, TransformerMixin
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

class AE_Wrapper(BaseEstimator, TransformerMixin):
    def __init__(self, x_shape, hidden_layers):
        self.x_shape= x_shape
        self.hidden_layers = hidden_layers
        self.autoencoder = None

    def create_autoencoder_model(self):
        input_layer = keras.layers.Input(shape=(self.x_shape,))
        self.encoded = input_layer
        for neuron in self.hidden_layers:
            self.encoded = keras.layers.Dense(neuron, activation='relu')(encoded)
        for neuron in reversed(self.hidden_layers):
            encoded = keras.layers.Dense(neuron, activation='relu')(encoded)
        decoded = keras.layers.Dense(self.x_shape, activation='sigmoid')(encoded)
        self.autoencoder = keras.Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return self.autoencoder

    def fit(self, X, y=None, **kwargs):
        self.clf_ = KerasClassifier(
            build_fn=self.create_autoencoder_model,
            **kwargs
        )
        self.clf_.fit(X, X)
        return self

    def transform(self, X):
        return self.clf_.predict(X)

    def fit_transform(self, X, y=None, **kwargs):
        self.clf_ = KerasClassifier(
            build_fn=self.create_autoencoder_model,
            **kwargs
        )
        self.clf_.fit(X, X)
        return self.clf_.predict(X)
    
    def get_encoder(self):
        return self.encoded