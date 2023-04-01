import random
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

tf.random.set_seed(15)
np.random.seed(15)
random.seed(15)

class Autoencoder:
    def __init__(self, input_shape, num_hidden_layers, num_nodes):
        self.num_hidden_layers = num_hidden_layers

        # Define the input layer
        inputs = keras.layers.Input(shape=input_shape)

        # Define the encoder layers
        encoded = inputs
        for i in range(self.num_hidden_layers):
            encoded = keras.layers.Dense(num_nodes[i], activation='relu')(encoded)

        # Define the decoder layers
        decoded = encoded
        for i in reversed(range(self.num_hidden_layers)):
            decoded = keras.layers.Dense(num_nodes[i], activation='relu')(decoded)
        decoded = keras.layers.Dense(input_shape[0], activation='sigmoid')(decoded)

        # Define the autoencoder model
        self.autoencoder = keras.models.Model(inputs=inputs, outputs=decoded)

        # Compile the autoencoder model
        self.autoencoder.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

        # Define the encoder model
        self.encoder = keras.models.Sequential()
        for i in range(self.num_hidden_layers):
            self.encoder.add(keras.layers.Dense(num_nodes[i], activation='relu', input_shape=input_shape))
        self.encoder.build(input_shape)

    def train(self, x_train, x_val, epochs, batch_size):
        '''
        Method to train the autoencoder by passing all relevant parameters
        to the fit method.
        '''
        history = self.autoencoder.fit(x_train, x_train,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       validation_data=(x_val, x_val))
        return history

    def get_encoder(self):
        '''
        Method that returns the encoder model.
        '''
        return self.encoder

    def get_decoder(self):
        '''
        Method that returns the decoder model.
        '''
        encoded_input = keras.layers.Input(shape=(self.encoder.output_shape[1],))
        decoder_layers = self.autoencoder.layers[self.num_hidden_layers+1:]
        decoded = encoded_input
        for layer in decoder_layers:
            decoded = layer(decoded)
        decoder = keras.models.Model(inputs=encoded_input, outputs=decoded)
        return decoder

    def plot_encoded_output(self, x_test, y_test=None, n_clusters=10):
        '''
        Method that plots the encoder's predictions on a scatter plot by
        clustering the comparison of each feature with the KMeans method.
        Returns a file of all generated plots.
        '''
        encoded_output = self.encoder.predict(x_test)
        n_features = encoded_output.shape[1]
        kmeans = KMeans(n_clusters=n_clusters, random_state=15).fit(encoded_output)
        labels = kmeans.labels_
        columns = x_test.columns
        encoded_df = pd.DataFrame(data=encoded_output, columns=['{}'.format(col) for col in columns])
        if y_test is not None:
            encoded_df['label'] = y_test
        sns.pairplot(encoded_df, hue='label', diag_kind='kde')
        plt.show()

    def cal_corr_coeff(self, x_test):
        '''
        Method that calculates the Perason's correlation coefficient for the
        comparison of each feature.
        Gives a data representation of the plots generated from the
        plot_encoded_output method.
        Returns a dataframe of float values.
        '''
        corr_coeffs = pd.DataFrame(columns=x_test.columns[:-1], index=x_test.columns[:-1])
        for i in range(len(corr_coeffs)):
            for j in range(i+1, len(corr_coeffs)):
                feature1 = corr_coeffs.columns[i]
                feature2 = corr_coeffs.columns[j]
                corr_coef = np.corrcoef(x_test[feature1], x_test[feature2])[0, 1]
                corr_coeffs.loc[feature1, feature2] = corr_coef
                corr_coeffs.loc[feature2, feature1] = corr_coef
        return corr_coeffs
    
    def top_10_coeffs(self, corr_coef):
        '''
        Returns the top 10 feature comparisons from the cal_corr_coeff method.
        Prints a string of the calculated value followed by the two correlated
        features.
        '''
        # Get the upper-triangle of the correlation matrix (excluding the diagonal)
        corr_coef_upper = np.triu(corr_coef, k=1)

        # Flatten the correlation matrix to a 1D array
        corr_coef_flat = corr_coef_upper.flatten()

        # Get the indices of the top 10 coefficients
        top_indices = np.argsort(corr_coef_flat)[::-1][:10]

        # Get the top 10 coefficients and their corresponding indices
        top_coef = corr_coef_flat[top_indices]
        top_indices_i, top_indices_j = np.unravel_index(top_indices, corr_coef_upper.shape)

        # Print the top 10 coefficients and their corresponding indices
        for coef, i, j in zip(top_coef, top_indices_i, top_indices_j):
            # print(f"Coefficient: {coef:.3f}, Index 1: url_{i}, Index 2: url_{j}")
            print(f"Coefficient: {coef:.3f}, Index 1: {corr_coef.columns[i]}, Index 2: {corr_coef.columns[j]}")