import pandas as pd
from sklearn.preprocessing import StandardScaler

class Processor():
    def __init__(self):
        #the artists considered in this dataset
        self.artists = []

        # the words / features
        self.words = []

        # the scalar for the dataframe
        self.scalar = StandardScaler()


    def fit_process(self, data, artists, num_features = 100):
        self.artists = artists

        data = data.loc[:, ['artist', 'song',"text"]]

        # Turn text lower case and remove punctuation
        data["text"] = data["text"].str.lower().str.replace('[^\w\s]','')
        data_copy = data.copy(deep=True) # save a copy of the data in solid string form
        data["text"] = data["text"].str.strip().str.split() #remove unnecessary spaces and turn into a list of words

        # Get the artist data set
        data_artists = self.subset_data(data_copy, artists)

        # Get the unique words in the artist data in a dataframe corresponding to the word counts
        word_count_data = data_artists.copy().text.str.split(expand=True).stack().value_counts().reset_index()
        word_count_data.columns = ['Word', 'Count']
        self.words =  word_count_data['Word'].tolist()[:num_features] # use the top num_features words as features

        # Add the unique words as features
        for word in self.words:
            data_artists[word] =  data_artists["text"].map(lambda x: x.count(word))
        
        X, y = data_artists.drop(['song', 'artist', 'text'], axis = 1), data_artists['artist']
        
        # Scale the data (and fit the scalar)
        X = pd.DataFrame(self.scalar.fit_transform(X), columns = X.columns)

        return X, y
    
    def process(self, data): #assumes that fit_process has been called
        data = data.loc[:, ['artist', 'song',"text"]]

        # Turn text lower case and remove punctuation
        data["text"] = data["text"].str.lower().str.replace('[^\w\s]','')
        data_copy = data.copy(deep=True) # save a copy of the data in solid string form
        data["text"] = data["text"].str.strip().str.split() #remove unnecessary spaces and turn into a list of words

        # Get the artist data set (using the existing list of artists)
        data_artists = self.subset_data(data_copy, self.artists)

        # Add the same features
        for word in self.words:
            data_artists[word] =  data_artists["text"].map(lambda x: x.count(word))

        X, y = data_artists.drop(['song', 'artist', 'text'], axis = 1), data_artists['artist']

        # Scale the data (and fit the scalar)
        X = pd.DataFrame(self.scalar.fit_transform(X), columns = X.columns)

        return X, y
    
    def subset_data(self, data_full, artists):
        """
        take a subset of the data that corresponds to the values in the list, classes
        """
        cols = data_full.columns
        array_data = data_full.to_numpy()

        # copy over data entries corresponding to the correct author
        subset_data = []
        for i in range(len(data_full)):
            artist = data_full['artist'].values[i]
            if artist in artists:
                subset_data.append(array_data[i])

        return pd.DataFrame(subset_data, columns = cols)