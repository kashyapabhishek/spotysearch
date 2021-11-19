import pandas as pd
import numpy as np
import os


class LoadData(object):

    def __init__(self):
        super().__init__()
        self.base_path = 'row_files'
        self.data = self.__load()

    def __load(self):
        return pd.read_csv(os.path.join(self.base_path, 'spotify_dataset.csv'), skipinitialspace=True, usecols=['artistname', 'trackname', 'playlistname'])

        

