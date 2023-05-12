import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import time

from CatBoostWrapper import CatBoostWrapper
from LightFMWrapper import LightFMWrapper
from NNWrapper import NNWrapper
from SurpriseWrapper import SurpriseWrapper



def rank_candidates_for_user(user: int, params: dict, candidates: dict, joke_quality:dict, joke_volume:dict, svd: SurpriseWrapper):
   
    n_users = 24983
    n_items = 100     
    user_id = user - 1
    rec_list = None
    
    if user_id < n_users:        
        res = {}
        for i, (model_name, value) in enumerate(candidates.items()):
            rank = {x: params[model_name] * (10 - j) for j, x in enumerate(value[user_id])}

            for k, v in rank.items():
                if k in res:
                    res[k] += v
                else:
                    res[k] = v
                    
        for k in res:
            res[k] += joke_quality[k] * params['quality']
            res[k] += joke_volume[k] * params['volume']

        rec_list = [x[0] for x in sorted(res.items(), key=lambda item: item[1], reverse=True)][:10]
        first_rating = svd.model.test([[user_id, rec_list[0], 0]])[0].est
        
    else:
        res = {}
                    
        for k in range(n_items):
            if joke_quality[k] < 0.00:# or joke_volume[k] < 0.1:
                continue
                
            res[k] = joke_quality[k] * params['quality']
            res[k] += joke_volume[k] * params['volume']

        xx = np.array(
            [[x[0], x[1]] for x in sorted(res.items(), key=lambda item: item[1], reverse=True)])
        
        pp = xx[:, 1]
        pp += np.abs(np.min(pp))
        rec_list = list(np.random.choice(xx[:, 0], size=10, replace=False, p=pp/np.sum(pp)))        
        first_rating = 0
            
    rec_list = [int(joke + 1) for joke in rec_list]
    return [{rec_list[0] : first_rating}, rec_list]
        
        
def solution(input_file_name: str, output_file_name: str):
    df = pd.read_csv('/magnit_recsys-in-practice/data/train_joke_df.csv') # Чтобы не рекомендовать айтемы, которые пользователь уже оценил    
    df['UID'] = df['UID'] - 1
    df['JID'] = df['JID'] - 1
    df = df.sort_values(by=['UID', 'JID'])
    df = df.reset_index(drop=True)
    
    with open('/magnit_recsys-in-practice/models/joke_quality.pkl', 'rb') as f: # Чем больше у айтема положительных оценок и меньше отрицательных - тем больше вес
        joke_quality = pickle.load(f)
        
    with open('/magnit_recsys-in-practice/models/joke_volume.pkl', 'rb') as f: # Шутки по середине будут дороже, чем по краям
        joke_volume = pickle.load(f)
       
    wrappers ={
        'svd':SurpriseWrapper(model_name='svd.surprise', folder_name='/magnit_recsys-in-practice/models', df_train=df),
        'catboost':CatBoostWrapper(model_name='catboost', folder_name='/magnit_recsys-in-practice/models', df_train=df),
        'nn_bias':NNWrapper(model_name='nn_bias.ckpt', folder_name='/magnit_recsys-in-practice/models', df_train=df)
    }
    
    candidates = {model_name: wrapper.predict() for model_name, wrapper in wrappers.items()}
    
    params = {'svd': 0.5, 'nn_bias': 0.1, 'catboost': -0.7, 'quality': -0.1, 'volume': 0.7} # Эмпирические коэффициенты    
    
    input_df = pd.read_csv(input_file_name)
    df_users = pd.DataFrame(np.unique(input_df['UID']), columns =['UID'])
    df_users['recommendations'] = df_users['UID'].apply(rank_candidates_for_user, args=(
                params,
                candidates,
                joke_quality,
                joke_volume,
                wrappers['svd']
            ))
    
    #output_df = input_df.merge(df_users)
    output_df = df_users
    
    output_df.to_csv(output_file_name, sep=';', index=False)
