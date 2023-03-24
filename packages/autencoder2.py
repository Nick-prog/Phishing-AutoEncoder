from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import CategoricalCrossentropy

class Autoencoder2:
    def __init__(self, input_dim, encoding_dim, num_hidden_layers):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.num_hidden_layers = num_hidden_layers
        self.autoencoder_model = self._build_autoencoder()
        self.encoder_model = self._build_encoder()
        self.decoder_model = self._build_decoder()
        self.loss = CategoricalCrossentropy()
    
    def _build_autoencoder(self):
        input_layer = Input(shape=(self.input_dim,))
        hidden_layer = input_layer
        for i in range(self.num_hidden_layers):
            hidden_layer = Dense(units=2**(i+1)*self.encoding_dim, activation='relu')(hidden_layer)
        bottleneck_layer = Dense(units=self.encoding_dim, activation='relu')(hidden_layer)
        hidden_layer = bottleneck_layer
        for i in range(self.num_hidden_layers):
            hidden_layer = Dense(units=2**(self.num_hidden_layers-i)*self.encoding_dim, activation='relu')(hidden_layer)
        output_layer = Dense(units=self.input_dim, activation='softmax')(hidden_layer)
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss="mse")
        return autoencoder
    
    def _build_encoder(self):
        input_layer = Input(shape=(self.input_dim,))
        hidden_layer = input_layer
        for i in range(self.num_hidden_layers):
            hidden_layer = Dense(units=2**(i+1)*self.encoding_dim, activation='relu')(hidden_layer)
        bottleneck_layer = Dense(units=self.encoding_dim, activation='relu')(hidden_layer)
        encoder = Model(inputs=input_layer, outputs=bottleneck_layer)
        return encoder
    
    def _build_decoder(self):
        bottleneck_layer = Input(shape=(self.encoding_dim,))
        hidden_layer = bottleneck_layer
        for i in range(self.num_hidden_layers):
            hidden_layer = Dense(units=2**(self.num_hidden_layers-i)*self.encoding_dim, activation='relu')(hidden_layer)
        output_layer = Dense(units=self.input_dim, activation='softmax')(hidden_layer)
        decoder = Model(inputs=bottleneck_layer, outputs=output_layer)
        return decoder
    
    def train(self, x_train, x_test, epochs, batch_size):
        self.autoencoder_model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))
    
    def get_encoder(self):
        return self.encoder_model
    
    def get_decoder(self):
        return self.decoder_model