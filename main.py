import packages
import numpy as np

def main():
    '''
    Function that runs the initial .csv dataset found from Phishtank.
    All features created and used were curated towards the infomation
    present in the original file.
    '''
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
    '''
    Function that tested the final version of the autoencoder we will be
    using throughout our experiment.
    The .csv dataset used contained features that can be found with any
    list of urls.
    Only contained 5 manually created features.
    '''
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
    autoencoder.plot_encoded_output(x_valid, y_valid)
    coeffs = autoencoder.cal_corr_coeff(x_valid)
    print(coeffs)

def main3():
    '''
    Function that tested the final version of our autoencoder with a larger
    dataset than normal.
    The .xlsx file combined the Kaggle and PhishTank datasets we originally had
    and created a couple new features.
    Only contained 9 manually created features.
    '''
    data = packages.Preprocess("C:/Users/Nick/Documents/Projects/Excel/combined_phish.xlsx", 'Sheet4')
    combined = data.download()
    
    full_dataset = combined.copy() # copy dataset.

    m_labels = packages.pd.DataFrame(full_dataset['label']) # Label only dataframe.

    # Feature only dataframe. Drop irrelavant columns and scale with mmx_scale method.
    m_features = full_dataset.drop(['url', 'label'] , axis=1)
    m_features_scaled = data.mmx_scale(m_features)

    # Pass feature and label dataframe to starify_shuffle_split method. Obtain a training and testing set for both.
    train_feature, train_label, test_feature, test_label = data.straify_shuffle_split(m_features_scaled, m_labels)
    
    # Separate our training data into a training and validation set.
    x_train, x_valid, y_train, y_valid = data.train_test_split(train_feature, train_label)
    
    # Initialize the autoencoder model
    num_hidden_layers = 3
    num_nodes = [36, 18, x_train.shape[1]] # List of how many neurons per layer
    input_shape = (x_train.shape[1],) # Based on number of features
    autoencoder = packages.Autoencoder(input_shape, num_hidden_layers, num_nodes)

    # Train the autoencoder model
    epochs = 1
    batch_size = 16
    autoencoder.train(x_train, x_train, epochs, batch_size)

    # print(autoencoder.get_encoder().summary())
    coeffs = autoencoder.cal_corr_coeff(x_valid)
    print(coeffs)
    autoencoder.top_10_coeffs(coeffs)
    # autoencoder.plot_encoded_output(x_valid, y_valid)

def main4():
    '''
    Function that tested the final version of our autoencoder.
    The .csv dataset used was the original pulled from PhishTank.
    We utilized the Word2Vec function from the gensim library to generate
    semantic based features from the urls themselves.
    Only contained 100 dynamically created features.
    '''
    data = packages.Preprocess("C:/Users/Nick/Documents/Projects/Excel/verified_online.csv")
    combined = data.download()
    
    full_dataset = combined.copy() # copy dataset.

    features = packages.pd.DataFrame(full_dataset['url'])
    labels = packages.pd.DataFrame(full_dataset['label'])

    encoded_features = data.word_2_vec(features, "url")
    # print(encoded_features.head())

    features_scaled = data.mmx_scale(encoded_features)

    # Pass feature and label dataframe to starify_shuffle_split method. Obtain a training and testing set for both.
    train_feature, train_label, test_feature, test_label = data.straify_shuffle_split(features_scaled, labels)

    # Separate our training data into a training and validation set.
    x_train, x_valid, y_train, y_valid = data.train_test_split(train_feature, train_label)
    
    # Initialize the autoencoder model
    num_hidden_layers = 3
    num_nodes = [x_train.shape[1]*3, x_train.shape[1]*2, x_train.shape[1]] # List of how many neurons per layer
    input_shape = (x_train.shape[1],) # Based on number of features
    autoencoder = packages.Autoencoder(input_shape, num_hidden_layers, num_nodes)

    # Train the autoencoder model
    epochs = 1
    batch_size = 256
    autoencoder.train(x_train, x_train, epochs, batch_size)

    coeffs = autoencoder.cal_corr_coeff(x_valid)
    print(coeffs)
    autoencoder.top_10_coeffs(coeffs)
    autoencoder.plot_encoded_output(x_valid, y_valid)


if __name__ == '__main__':
    main4()