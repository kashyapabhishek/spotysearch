import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing.preprocess import Preprocess
import os

class ModelBuilder(object):

    def __init__(self):
        super().__init__()
        self.data = Preprocess().preprocess_pipeline()
        self.base_path_for_row_files = 'row_files'
        self.base_path_for_model = 'models'
        self.vectorizer = None
        self.y = None
        self.X = None
        self.cosine_similarities = None
        self.document_scores = None
        self.related_docs_indices = None
        self.dic_df = self.__load_dictonary()
        self.vector_model = self.__load_model()
        self.clean_final_data = self.__load_clean_final_data()
        

    def build_model(self): 
        self.vectorizer = TfidfVectorizer(vocabulary=self.dic_df['feature'].tolist())
        self.X = self.vectorizer.fit_transform(self.data['text'])

    def __load_dictonary(self):
        self.dic_df = pd.read_csv(os.path.join(self.base_path_for_row_files, 'word_dic.csv'))
        return self.dic_df

    



        


