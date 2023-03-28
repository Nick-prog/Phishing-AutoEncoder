import pandas as pd
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.model_selection import train_test_split as TTS

class Preprocess:
    '''
    Class created for all preprocessing data purposes.
    '''

    def __init__(self, path, name="Sheet1"):
        self.path = path
        self.name = name
        self.random_state = 16

    def download(self):
        if(self.name == "Sheet1"):
            data = pd.read_csv(self.path, verbose=1)
        else:
            data = pd.read_excel(self.path, sheet_name=self.name, verbose=1)
        self.df = pd.DataFrame(data)
        return self.df
    
    def mmx_scale(self, df):
        scaled_array = MMS().fit_transform(df)
        return pd.DataFrame(scaled_array, columns=df.columns)
    
    def straify_shuffle_split(self, feature, label):
        split = SSS(n_splits=3, test_size=0.2, random_state=self.random_state)
        for train_index, test_index in split.split(feature, label):
            train_feature, train_label = feature.iloc[train_index], label.iloc[train_index]
            test_feature, test_label = feature.iloc[test_index], label.iloc[test_index]
        return train_feature, train_label, test_feature, test_label
    
    def train_test_split(self, feature, label):
        x_train, x_valid, y_train, y_valid = TTS(feature, label, test_size=0.2, random_state=self.random_state)
        return x_train, x_valid, y_train, y_valid

    
if __name__ == "__main__":
    
    # Path way for downloading .csv file. Passed to Preprocess class.
    data = Preprocess("C:/Users/Nick/Documents/Projects/Excel/verified_online.csv")
    phishtank = data.download()

    full_dataset = phishtank.copy() # copy dataset.
    print("Length of full dataset:", len(full_dataset), "\n")

    pt_labels = pd.DataFrame(full_dataset['label']) # Label only dataframe.

    # Feature only dataframe. Drop irrelavant columns and scale with mmx_scale method.
    pt_features = full_dataset.drop(['phish_id', 'url', 'phish_detail_url', 'submission_time', 'verified',
                                  'verification_time', 'online', 'target', 'protocol', 'top_lvl_domain', 'label'] , axis=1)
    pt_features_scaled = data.mmx_scale(pt_features)

    # Pass feature and label dataframe to starify_shuffle_split method. Obtain a training and testing set for both.
    train_feature, train_label, test_feature, test_label = data.straify_shuffle_split(pt_features_scaled, pt_labels)
    print("Length of training features:", len(train_feature))
    print("Length of training labels:", len(train_label))

    print("Length of testing features:", len(test_feature))
    print("Length of testing labels:", len(test_label))

    # data = Preprocess("C:/Users/Nick/Documents/Projects/Excel/malicious_phish.csv")
    # kaggle1 = data.download()

    # data = Preprocess("C:/Users/Nick/Documents/Projects/Excel/phishing_site_urls.csv")
    # kaggle2 = data.download()
