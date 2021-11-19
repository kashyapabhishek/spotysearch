from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np
import pickle
import os

class Predict(object):

    def __init__(self):
        super().__init__()
        self.base_path_for_row_files = 'row_files'
        self.base_path_for_model = 'models'
        self.y = None
        self.cosine_similarities = None
        self.document_scores = None
        self.related_docs_indices = None
        self.dic_df = self.__load_dictonary()
        self.vector_model = self.__load_model()
        self.clean_final_data = self.__load_clean_final_data()

    def __load_dictonary(self):
        self.dic_df = pd.read_csv(os.path.join(self.base_path_for_row_files, 'word_dic.csv'))
        return self.dic_df

    def __load_model(self):
        filename = os.path.join(self.base_path_for_model, 'finalized_model.sav')
        self.vector_model = pickle.load(open(filename, 'rb'))
        return self.vector_model

    def __load_clean_final_data(self):
        self.clean_final_data = pd.read_csv(os.path.join(self.base_path_for_row_files, 'clean_final_data.csv'))

    def __convert_to_vector(self, querys):
        vectorizer = TfidfVectorizer(vocabulary=self.dic_df['feature'].tolist())
        return vectorizer.fit_transform(querys)

    def search(self, querys):
        return_result = []
        if len(querys) > 0:
            self.y = self.__convert_to_vector(querys)
            self.cosine_similarities = linear_kernel(self.y, self.vector_model).flatten()
            #self.document_scores = [item.item() for item in self.cosine_similarities[1:]]
            self.related_docs_indices = self.cosine_similarities.argsort()[:-5:-1]
            self.clean_final_data = None
            if self.clean_final_data == None:
                self.clean_final_data = pd.read_csv(os.path.join(self.base_path_for_row_files, 'clean_final_data.csv'))
            for i in self.related_docs_indices:
                return_result.append(self.clean_final_data['text'].iloc[i])
        return return_result
