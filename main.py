import time
import packages
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasClassifier

random.seed(15)
np.random.seed(15)

def syntactic_dataset():
    '''
    Function just to test the dataset used by one of our references.
    Need to replicate a similar number of features and labels for the semantic dataset.
    '''
    data = packages.Preprocess("C:/Users/Nick/Documents/Projects/Excel/FinalDataset/All.csv")
    combined = data.download()
    
    full_dataset = combined.copy() # copy dataset.
    full_dataset = full_dataset.dropna()

    features = full_dataset.drop(['URL_Type_obf_Type', 'label'] , axis=1)
    labels = packages.pd.DataFrame(full_dataset['label'])
    print("Positive samples:", (labels == 1).sum().sum(), "Percentage: ", ((labels == 1).sum().sum())/len(labels)*100)
    print("Negative samples:", (labels == 0).sum().sum(), "Percentage: ", ((labels == 0).sum().sum())/len(labels)*100)
    print("Total:", len(labels))

def semantic_dataset():
    '''
    Function to run our semantic dataset approach through our created AE and classifiers.
    Need to run:
    - Try running with different number of hidden layers (1-5)
    - Try running with more epochs (100-500)
    - Trying different hyper parameters (activation functions, optimizers, neurons, etc...)
    Always consider the possibiltiy of over training, if results are similar don't push it too much farther.
    '''
    data = packages.Preprocess("C:/Users/Nicko/Documents/Cyber INT/Phishing-AutoEncoder/test.csv")
    combined = data.download()
    
    full_dataset = combined.copy() # copy dataset.
    full_dataset = full_dataset.drop(full_dataset[full_dataset['label'] == 1].sample(min((full_dataset['label'] == 1).sum(), 6428)).index)
    full_dataset = full_dataset.drop(full_dataset[full_dataset['label'] == 0].sample(min((full_dataset['label'] == 0).sum(), 11297)).index)

    features = packages.pd.DataFrame(full_dataset['url'])
    labels = packages.pd.DataFrame(full_dataset['label'])
    print("Positive samples:", (labels == 1).sum().sum(), "Percentage: ", ((labels == 1).sum().sum())/len(labels)*100)
    print("Negative samples:", (labels == 0).sum().sum(), "Percentage: ", ((labels == 0).sum().sum())/len(labels)*100)
    print("Total:", len(labels))

    encoded_features = data.word_2_vec(features, "url")

    features_scaled = data.mmx_scale(encoded_features)

    # Pass feature and label dataframe to starify_shuffle_split method. Obtain a training and testing set for both.
    train_feature, train_label, test_feature, test_label = data.straify_shuffle_split(features_scaled, labels)

    # Separate our training data into a training and validation set.
    x_train, x_valid, y_train, y_valid = data.train_test_split(train_feature, train_label)

    hidden = 5
    epoch = 100
    nodes = [x_train.shape[1] * (hidden-i) for i in range(hidden)]
    active = "relu"
    # active = tf.keras.layers.LeakyReLU(alpha=0.3)

    # Initialize the autoencoder model
    autoencoder = packages.Autoencoder(input_shape = (x_train.shape[1],), # Based on number of features
                                       num_hidden_layers = hidden, 
                                       num_nodes = nodes,
                                       active = active) # List of how many neurons per layer
    
    # Train the autoencoder model
    history = autoencoder.train(x_train, x_valid, epochs=epoch, batch_size=256)

    encoder = autoencoder.get_encoder()
    x_train_pred = encoder.predict(x_train)
    x_valid_pred = encoder.predict(x_valid)

    classifier = packages.Classifier(x_train_pred, x_valid_pred, y_train, y_valid, test_feature, test_label)
    classifier.decision_tree(depth=5)
    classifier.logistic_regression(max_iter=1000)
    classifier.random_forest(estimators=100)
    classifier.support_vector_machine(kernel="linear", regularization=1)
    print(f"{epoch} epochs with {hidden} hidden layers for the {active} activation function. End.")

if __name__ == '__main__':
    # syntactic_dataset()
    start_time = time.time()

    semantic_dataset()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")