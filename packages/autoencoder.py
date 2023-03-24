from tensorflow import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

class Autoencoder:
    def __init__(self, input_shape, num_hidden_layers, num_nodes):
        self.num_hidden_layers = num_hidden_layers

        # Define the input layer
        inputs = keras.layers.Input(shape=input_shape)

        # Define the encoder layers
        encoded = inputs
        for i in range(num_hidden_layers):
            encoded = keras.layers.Dense(num_nodes[i], activation='relu')(encoded)

        # Define the decoder layers
        decoded = encoded
        for i in reversed(range(num_hidden_layers)):
            decoded = keras.layers.Dense(num_nodes[i], activation='relu')(decoded)
        decoded = keras.layers.Dense(input_shape[0], activation='sigmoid')(decoded)

        # Define the autoencoder model
        self.autoencoder = keras.models.Model(inputs=inputs, outputs=decoded)

        # Compile the autoencoder model
        self.autoencoder.compile(optimizer='adam', loss="mse")

        # Define the encoder model
        self.encoder = keras.models.Sequential()
        for i in range(num_hidden_layers):
            self.encoder.add(keras.layers.Dense(num_nodes[i], activation='relu', input_shape=input_shape))
        self.encoder.build(input_shape)

    def train(self, x_train, x_val, epochs, batch_size):
        # Train the autoencoder model
        self.autoencoder.fit(x_train, x_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(x_val, x_val))

    def get_encoder(self):
        # Return the encoder model
        return self.encoder

    def get_decoder(self):
        # Define the decoder model
        encoded_input = keras.layers.Input(shape=(self.encoder.output_shape[1],))
        decoder_layers = self.autoencoder.layers[self.num_hidden_layers+1:]
        decoded = encoded_input
        for layer in decoder_layers:
            decoded = layer(decoded)
        decoder = keras.models.Model(inputs=encoded_input, outputs=decoded)
        return decoder

    def plot_encoded_output(self, x_test, y_test=None, n_clusters=10):
        encoded_output = self.encoder.predict(x_test)
        n_features = encoded_output.shape[1]
        kmeans = KMeans(n_clusters=n_clusters, random_state=15).fit(encoded_output)
        labels = kmeans.labels_
        columns = x_test.columns
        encoded_df = pd.DataFrame(data=encoded_output, columns=['encoded_{}'.format(col) for col in columns])
        if y_test is not None:
            encoded_df['label'] = y_test
        sns.pairplot(encoded_df, hue='label', diag_kind='kde')
        plt.show()

    def calculate_correlation_coefficients(self, x_test):
        corr_coeffs = pd.DataFrame(columns=x_test.columns[:-1], index=x_test.columns[:-1])
        for i in range(len(corr_coeffs)):
            for j in range(i+1, len(corr_coeffs)):
                feature1 = corr_coeffs.columns[i]
                feature2 = corr_coeffs.columns[j]
                corr_coef = np.corrcoef(x_test[feature1], x_test[feature2])[0, 1]
                corr_coeffs.loc[feature1, feature2] = corr_coef
                corr_coeffs.loc[feature2, feature1] = corr_coef
        return corr_coeffs