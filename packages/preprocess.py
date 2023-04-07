import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from sklearn.model_selection import train_test_split as TTS
from sklearn.preprocessing import OneHotEncoder

class Preprocess:
    '''
    Class created for all preprocessing data purposes.
    '''

    def __init__(self, path, name="Sheet1"):
        self.path = path
        self.name = name
        self.random_state = 16

    def download(self):
        '''
        Method that returns a dataframe of values captured from a downloded
        .csv or .xlsx file.
        '''
        if(self.name == "Sheet1"):
            data = pd.read_csv(self.path, verbose=1,)
        else:
            data = pd.read_excel(self.path, sheet_name=self.name, verbose=1)
        self.df = pd.DataFrame(data)
        return self.df
    
    def mmx_scale(self, df):
        '''
        Method that returns a dataframe of values that have been scaled between
        0 and 1 with the MinMaxScaler method.
        Same column names are passed for readability.
        '''
        scaled_array = MMS().fit_transform(df)
        return pd.DataFrame(scaled_array, columns=df.columns)
    
    def straify_shuffle_split(self, feature, label):
        '''
        Method that returns four dataframes that contain a 80/20 split of the passed
        feature and label datasets for training and testing purposes.
        Random stat set for replication capability.
        '''
        split = SSS(n_splits=3, test_size=0.2, random_state=self.random_state)
        for train_index, test_index in split.split(feature, label):
            train_feature, train_label = feature.iloc[train_index], label.iloc[train_index]
            test_feature, test_label = feature.iloc[test_index], label.iloc[test_index]
        return train_feature, train_label, test_feature, test_label
    
    def train_test_split(self, feature, label):
        '''
        Method that returns four dataframes that contain a 80/20 split of the passed
        feature and label dataframes for training purposes. 
        Random state set for replication capability.
        '''
        x_train, x_valid, y_train, y_valid = TTS(feature, label, test_size=0.2, random_state=self.random_state)
        return x_train, x_valid, y_train, y_valid
    
    def one_hot_encoder(self, df, column):
        '''
        Method that returns a dataframe of values represented as integer values
        based on the url string encodings generated through the OneHotEncoder library.
        '''
        top_categories = df[column].value_counts().nlargest(100).index.tolist()
        df_top_categories = df[df[column].isin(top_categories)]
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(df_top_categories[[column]])
        encoded_features = encoder.transform(df_top_categories[[column]]).toarray()
        new_column_names = [column + "_" + feature for feature in encoder.get_feature_names()]
        encoded_df = pd.DataFrame(encoded_features, columns=new_column_names)
        return encoded_df

        # encoder = OneHotEncoder(handle_unknown="ignore")
        # encoder_array = encoder.fit_transform(df).toarray()
        # encoded_df = pd.DataFrame(encoder_array, columns=encoder.get_feature_names_out([column]))
        # return encoded_df

    def feature_extraction(self, df, column):
        '''
        Method that returns a dataframe of values represented as integer values
        based word frequency found through generated vectors from the CountVectorizer library.
        '''
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer()
        vectorizer.fit(df[column])
        encoded_features = vectorizer.transform(df[column]).toarray()
        new_column_names = [column + "_" + feature for feature in vectorizer.get_feature_names()]
        encoded_df = pd.DataFrame(encoded_features, columns=new_column_names)
        return encoded_df
    
    def word_2_vec(self, df, column):
        '''
        Method that returns a dataframe of values represented as float values
        based on generated semantic relationship through the Word2Vec library.
        '''
        from gensim.models import Word2Vec
        sentences = [url.split("/") for url in df[column]]

        model = Word2Vec(sentences=sentences, vector_size=80, window=5, min_count=1, workers=4)

        # Transform URLs into embeddings using the trained Word2Vec model
        url_embeddings = []
        for url in sentences:
            embeddings = []
            for word in url:
                try:
                    embedding = model.wv[word]
                    embeddings.append(embedding)
                except KeyError:
                    pass
            url_embedding = np.mean(embeddings, axis=0)
            url_embeddings.append(url_embedding)

        # Create a new DataFrame with the encoded features
        encoded_df = pd.DataFrame(url_embeddings, columns=[f"{column}_{i}" for i in range(80)])
        return encoded_df
