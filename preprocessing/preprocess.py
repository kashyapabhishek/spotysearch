from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from datalayer.load_dataset import LoadData
import pandas as pd
import string
import nltk
import os

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class Preprocess(object):

    def __init__(self):
        super().__init__()
        self.data = None
        self.base_path = 'row_files'

    def concat_data_to_one_column(self):

        self.data['text'] = self.data['artistname'].astype(str) + self.data['trackname'].astype(str) 
        + self.data['playlistname'].astype(str)

        self.data.drop(['artistname', 'trackname', 'playlistname'], axis=1, inplace=True)

    def convert_to_lower(self):
        self.data['text'] = self.data['text'].apply(lambda x: x.lower())
    
    def remove_punctuation(self):
        self.data['text'] = self.data['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    def tokenize_string(self):
        self.data['text'] = self.data['text'].apply(lambda x: word_tokenize(x))

    def remove_stopwords(self):
        stw = stopwords.words('english')
        self.data['text'] = self.data['text'].apply(lambda x: [i for i in x if i not in stw])

    def lemmatize_data(self):
        lemmatizer = WordNetLemmatizer()
        self.data['text'] = self.data['text'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

    def clean_data(self):
        c_data = []
        for i in range(0, self.data.shape[0]):
            c_data.append("".join(self.data['text'].iloc[i][1:-1].split(',')).replace("'", ""))
            if i % 10000 == 0:
                print(i)

        self.data = pd.DataFrame(c_data, columns=['text'])

    def dump_clean_data(self):
        self.data.to_csv(os.path.join(self.base_path, 'clean_final_data.csv'), index=False)

    def preprocess_pipeline(self):
        obj = LoadData()
        self.data = obj.data
        self.concat_data_to_one_column()
        self.convert_to_lower()
        self.remove_punctuation()
        self.tokenize_string()
        self.lemmatize_data()
        self.clean_data()
        self.dump_clean_data()
        return self.data





        