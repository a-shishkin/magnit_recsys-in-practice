import pandas as pd
from typing import List

class Helper:   
    def create_test_set(self, n_users:int, n_items:int) -> List[tuple]:
        test_set = []
        for u in range(n_users):
            for j in range(n_items):
                test_set.append((u, j, 0))
                
        return test_set
        
        
    def filter_viewed_items(self, 
                            df_train: pd.DataFrame, 
                            df_pred: pd.DataFrame, 
                            n_users: int, 
                            n_recommendations: int,
                           ascending:bool=False) -> pd.DataFrame:

        
        df_train['UID'] = df_train['UID'].astype(int)
        df_train['JID'] = df_train['JID'].astype(int)
        
        
        df_pred['UID'] = df_pred['UID'].astype(int)
        df_pred['JID'] = df_pred['JID'].astype(int)
        
        mrg = df_pred.merge(df_train, on=["UID", "JID"], how="left", indicator=True)
        
        mrg = mrg[mrg['_merge'] == 'left_only']

        frames = []
        for user_id in range(n_users):
            recommended_items = mrg[mrg['UID'] == user_id]
            #display(recommended_items)
            recommended_items = recommended_items.sort_values('Rating_pred', ascending=ascending)
            frames.append(recommended_items.iloc[:n_recommendations])

        df_rec = pd.concat(frames).reset_index()
        df_rec = df_rec.drop(columns=['index'])
        
        res = list(df_rec.groupby('UID').agg({'JID':list})['JID'].values)
        
        return res