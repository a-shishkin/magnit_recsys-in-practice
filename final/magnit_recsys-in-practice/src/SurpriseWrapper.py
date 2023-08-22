import pandas as pd
from typing import List
from Helper import Helper
import os
import time

from surprise import dump, Reader


class SurpriseWrapper():
    def __init__(self,  model_name:str, folder_name:str, df_train: pd.DataFrame, logger, n_recommendations:int = 10, n_users:int = 24983, n_items = 100):
        self.n_users = n_users
        self.n_items = n_items
        self.model_name = model_name
        self.logger = logger
        self.reader = Reader(rating_scale=(-10, 10))
        _, self.model = dump.load(os.path.join(folder_name, model_name))
        self.n_recommendations = n_recommendations
        self.df_train = df_train
        self.helper = Helper()

    def predict(self) -> List[list]:
        t1 = time.time()
        test_set = self.helper.create_test_set(self.n_users, self.n_items)
                
        #print(len(test_set))
        predictions = self.model.test(test_set)
        #print(len(predictions))
        df_pred = pd.DataFrame([(x[0], x[1], x[3]) for x in predictions], columns = ['UID', 'JID', 'Rating_pred'])
        
        #display(df_pred)
        
        res = self.helper.filter_viewed_items(self.df_train, df_pred, self.n_users, self.n_recommendations)
        self.logger.info(f'{self.model_name} done; time: {round(time.time() - t1, 3)}')
        return res