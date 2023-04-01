import packages
import random
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasClassifier

random.seed(15)
np.random.seed(15)

def main():
    data = packages.Preprocess("C:/Users/Nick/Documents/Projects/Excel/test.csv")
    combined = data.download()
    
    full_dataset = combined.copy() # copy dataset.

    features = packages.pd.DataFrame(full_dataset['url'])
    labels = packages.pd.DataFrame(full_dataset['label'])

    encoded_features = data.word_2_vec(features, "url")

    features_scaled = data.mmx_scale(encoded_features)

    # Pass feature and label dataframe to starify_shuffle_split method. Obtain a training and testing set for both.
    train_feature, train_label, test_feature, test_label = data.straify_shuffle_split(features_scaled, labels)

    # Separate our training data into a training and validation set.
    x_train, x_valid, y_train, y_valid = data.train_test_split(train_feature, train_label)

    # Initialize the autoencoder model
    autoencoder = packages.Autoencoder(input_shape = (x_train.shape[1],), # Based on number of features
                                       num_hidden_layers = 3, 
                                       num_nodes = [x_train.shape[1]*3, x_train.shape[1]*2, x_train.shape[1]]) # List of how many neurons per layer
    
    # Train the autoencoder model
    history = autoencoder.train(x_train, x_valid, epochs=1, batch_size=256)

    encoder = autoencoder.get_encoder()
    x_train_pred = encoder.predict(x_train)
    x_valid_pred = encoder.predict(x_valid)

    classifier = packages.Classifier(x_train_pred, x_valid_pred, y_train, y_valid, test_feature, test_label)
    classifier.decision_tree(depth=5)
    classifier.logistic_regression(max_iter=1000)
    classifier.support_vector_machine(kernel="linear", regularization=1)
    classifier.random_forest(estimators=100)

    # autoencoder_model = KerasClassifier(model=packages.Autoencoder(input_shape=([x_train.shape[1]],), num_nodes=x_train.shape[1]), verbose=0)
    
    # # Initialize the autoencoder model
    # param_dist = { 
    #     'num_nodes': [x_train.shape[1]*3, x_train.shape[1]*2, x_train.shape[1]]
    # }

    # random_search = RandomizedSearchCV(autoencoder_model, param_distributions=param_dist,
    #                                    n_iter=3, cv=3)
    # random_search.fit(x_train, x_train, epochs=1, batch_size=36)

    # # Print the best hyperparameters and associated validation score
    # print(f"Best score: {random_search.best_score_}")
    # print(f"Best params: {random_search.best_params_}")

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

    features_scaled = data.mmx_scale(encoded_features)

    # Pass feature and label dataframe to starify_shuffle_split method. Obtain a training and testing set for both.
    train_feature, train_label, test_feature, test_label = data.straify_shuffle_split(features_scaled, labels)

    # Separate our training data into a training and validation set.
    x_train, x_valid, y_train, y_valid = data.train_test_split(train_feature, train_label)

    model = packages.AE_Wrapper(x_train.shape[1], [x_train.shape[1]*3, x_train.shape[1]*2, x_train.shape[1]])

    history = model.fit(x_train, epoch=1, batch_size=256, random_state=15)
    print(history.history['accuracy'][-1], history.history['val_accuracy'][-1])

    # coeffs = autoencoder.cal_corr_coeff(x_valid)
    # print(coeffs)
    # autoencoder.top_10_coeffs(coeffs)
    # autoencoder.plot_encoded_output(x_valid, y_valid)

if __name__ == '__main__':
    main()