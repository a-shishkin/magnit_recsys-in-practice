import pandas as pd
from typing import List
from Helper import Helper
import os
import time

from catboost import CatBoostRanker, Pool



class CatBoostWrapper():
    def __init__(self, model_name:str, folder_name:str, df_train: pd.DataFrame, n_recommendations:int = 10, n_users:int = 24983, n_items = 100):
        self.n_users = n_users
        self.n_items = n_items
        self.model_name = model_name
        
        self.model = CatBoostRanker()
        self.model.load_model(os.path.join(folder_name, model_name))
                
        self.n_recommendations = n_recommendations
        self.df_train = df_train.copy()
        self.helper = Helper()
        
        
    def predict(self) -> List[list]:
        t1 = time.time()
        test_set = self.helper.create_test_set(self.n_users, self.n_items)                
        df_test = pd.DataFrame(test_set, columns=['UID', 'JID', 'Rating'])
        test_pool = Pool(df_test, group_id=df_test['UID'], cat_features=['UID', 'JID'])        

        predictions = self.model.predict(test_pool)
        df_test['Rating_pred'] = predictions
        
        res = self.helper.filter_viewed_items(self.df_train, df_test, self.n_users, self.n_recommendations)        
        print(self.model_name, 'time:', round(time.time() - t1, 3))
        return res
            