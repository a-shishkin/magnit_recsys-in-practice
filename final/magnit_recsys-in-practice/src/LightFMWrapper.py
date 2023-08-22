import pandas as pd
from typing import List
from Helper import Helper
import os
import time

import numpy as np
import typing as tp
from lightfm import LightFM
from scipy.sparse import coo_matrix
from lightfm.data import Dataset as LFMDataset
import pickle
from sklearn.preprocessing import normalize


class LightFMWrapper():
    def __init__(self, model_name:str, folder_name:str, df_train: pd.DataFrame, n_recommendations:int = 10, n_users:int = 24983, n_items = 100):
        self.n_users = n_users
        self.n_items = n_items
        self.model_name = 'lightfm'
        
        with open(os.path.join(folder_name, '05_jokes_lfm_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(folder_name, '05_jokes_lfm_dataset.pkl'), 'rb') as f:
            self.lfm_dataset = pickle.load(f)
    
    
        self.train_matrix, _ = self.lfm_dataset.build_interactions(zip(*df_train[["UID", "JID"]].values.T))
    
        self.model.item_biases = np.zeros_like(self.model.item_biases)
        self.model.user_biases = np.zeros_like(self.model.user_biases)

        self.model.item_embeddings = normalize(self.model.item_embeddings)
        self.model.user_embeddings = normalize(self.model.user_embeddings)

        self.user2idx = self.lfm_dataset._user_id_mapping
        self.item2idx = self.lfm_dataset._item_id_mapping
        self.idx2item = {v:k for k,v in self.item2idx.items()}
        self.idx2user = {v:k for k,v in self.user2idx.items()}
                
        self.n_recommendations = n_recommendations
        self.df_train = df_train.copy()
        self.helper = Helper()
        
    def lfm_get_n_recommendations_for_user(self,
            user_id: str,
            model: LightFM,
            train_matrix: coo_matrix,
            user_to_id: tp.Dict[str, int],
            id_to_item: tp.Dict[int, str],
            n_recommendations: int
        ) -> pd.DataFrame:


        user_inner_id = user_to_id[user_id]
        scores = model.predict(user_ids=user_inner_id, item_ids=np.arange(train_matrix.shape[1]))
        user_watched_items = train_matrix.col[train_matrix.row == user_inner_id]
        scores[user_watched_items] = -np.inf

        recommended_item_inner_ids = np.argpartition(scores, -np.arange(n_recommendations))[-n_recommendations:][::-1]
        recommended_item_ids = [id_to_item[x] for x in recommended_item_inner_ids]
        return recommended_item_ids


    def predict(self) -> List[list]:
        t1 = time.time()
        test_set = self.helper.create_test_set(self.n_users, self.n_items)                
        df_test = pd.DataFrame(test_set, columns=['UID', 'JID', 'Rating'])   

        recommendations = pd.DataFrame({"UID": df_test["UID"].unique()})
        recommendations["JID"] = recommendations["UID"].apply(
            self.lfm_get_n_recommendations_for_user,
            args=(
                self.model,
                self.train_matrix,
                self.user2idx,
                self.idx2item,
                self.n_recommendations
            ),
        )
        recommendations = recommendations.explode("JID")
        recommendations["Rating_pred"] = recommendations.groupby(["UID"]).cumcount() + 1

        res = self.helper.filter_viewed_items(self.df_train, recommendations, self.n_users, self.n_recommendations, ascending=True)   
        print(self.model_name, 'time:', round(time.time() - t1, 3))     
        return res
            