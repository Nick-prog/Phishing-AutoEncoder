import packages
import numpy as np

def main():
    # Path way for downloading .csv file. Passed to Preprocess class.
    data = packages.Preprocess("C:/Users/Nick/Documents/Projects/Excel/verified_online.csv")
    phishtank = data.download()

    full_dataset = phishtank.copy() # copy dataset.

    pt_labels = packages.pd.DataFrame(full_dataset['label']) # Label only dataframe.

    # Feature only dataframe. Drop irrelavant columns and scale with mmx_scale method.
    pt_features = full_dataset.drop(['phish_id', 'url', 'phish_detail_url', 'submission_time', 'verified',
                                  'verification_time', 'online', 'target', 'protocol', 'top_lvl_domain', 'label'] , axis=1)
    pt_features_scaled = data.mmx_scale(pt_features)

    # Pass feature and label dataframe to starify_shuffle_split method. Obtain a training and testing set for both.
    train_feature, train_label, test_feature, test_label = data.straify_shuffle_split(pt_features_scaled, pt_labels)
    
    # Separate our training data into a training and validation set.
    x_train, x_valid, y_train, y_valid = data.train_test_split(train_feature, train_label)

    # # Initialize the autoencoder model
    # num_hidden_layers = 1
    # num_nodes = [5]
    # input_shape = (10,)
    # autoencoder = packages.Autoencoder(input_shape, num_hidden_layers, num_nodes)

    # # Train the autoencoder model
    # epochs = 10
    # batch_size = 16
    # autoencoder.train(x_train, x_valid, epochs, batch_size)

    # Train the autoencoder
    autoencoder = packages.Autoencoder2(input_dim=10, encoding_dim=5, num_hidden_layers=1)
    autoencoder.train(x_train, x_valid, epochs=50, batch_size=256)

def main2():
    data = packages.Preprocess("C:/Users/Nick/Documents/Projects/Excel/malicious_phish.csv")
    malicious = data.download()

    full_dataset = malicious.copy() # copy dataset.

    m_labels = packages.pd.DataFrame(full_dataset['label']) # Label only dataframe.

    # Feature only dataframe. Drop irrelavant columns and scale with mmx_scale method.
    m_features = full_dataset.drop(['url', 'type', 'label'] , axis=1)
    m_features_scaled = data.mmx_scale(m_features)

    # Pass feature and label dataframe to starify_shuffle_split method. Obtain a training and testing set for both.
    train_feature, train_label, test_feature, test_label = data.straify_shuffle_split(m_features_scaled, m_labels)
    
    # Separate our training data into a training and validation set.
    x_train, x_valid, y_train, y_valid = data.train_test_split(train_feature, train_label)
    
    # Initialize the autoencoder model
    num_hidden_layers = 2
    num_nodes = [10, 5] # List of how many neurons per layer
    input_shape = (5,) # Based on number of features
    autoencoder = packages.Autoencoder(input_shape, num_hidden_layers, num_nodes)

    # Train the autoencoder model
    epochs = 1
    batch_size = 16
    autoencoder.train(x_train, x_train, epochs, batch_size)

    # print(autoencoder.get_encoder().summary())
    # autoencoder.plot_encoded_output(x_valid, y_valid)
    coeffs = autoencoder.calculate_correlation_coefficients(x_valid)
    print(coeffs)

if __name__ == '__main__':
    main2()