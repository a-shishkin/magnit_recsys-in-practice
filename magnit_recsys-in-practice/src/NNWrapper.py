import pandas as pd
from typing import List
from Helper import Helper
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as td
import pytorch_lightning as pl


class ContextualRanker(pl.LightningModule):
    def __init__(self, embedding_dim, n_users, n_items, ratings_range):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.y_range = ratings_range
        
        self.user_embedding = nn.Embedding(n_users+1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(n_items+1, embedding_dim, padding_idx=0)

        self.user_bias = nn.Embedding(n_users+1, 1, padding_idx=0)
        self.item_bias = nn.Embedding(n_items+1, 1, padding_idx=0)

    def forward(self, x):
        users, items = x[ : , 0], x[ : , 1]
        dot = self.user_embedding(users) * self.item_embedding(items)
        result = dot.sum(1)
        result = (result + self.user_bias(users).squeeze() + self.item_bias(items).squeeze())
        return (torch.sigmoid(result) * (self.y_range[1] - self.y_range[0]) + self.y_range[0])
    
    def step(self, batch, batch_idx, metric, prog_bar=False):
        x, y = batch
        predictions = self.forward(x)
        loss = RMSE_loss(predictions, y.float())
        self.log(metric, loss, prog_bar=prog_bar)
        return loss

    def test_step(self, batch, batch_idx, prog_bar=False):
        return self.step(batch, batch_idx, "test_loss")

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train_loss")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val_loss", True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]
    
    
class ContextualRankerData(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, features):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.features = features

        
    def prepare_data(self):
        self.test_data = self.test_data.assign(rdm = np.random.random(len(self.test_data)))

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = td.TensorDataset(
              torch.from_numpy(self.train_data[self.features].values), 
              torch.from_numpy(self.train_data["Rating"].values)
              )
  
            self.val_dataset = td.TensorDataset(
              torch.from_numpy(self.val_data[self.features].values), 
              torch.from_numpy(self.val_data["Rating"].values)
              )
          
        if stage == "test" or stage is None:  
            self.test_dataset = td.TensorDataset(
                torch.from_numpy(self.test_data[self.features].values),
                torch.from_numpy(self.test_data[["rdm"]].values)
            )
            
    def train_dataloader(self):
        return td.DataLoader(self.train_dataset, batch_size=2048, shuffle=True, num_workers=0)
  
    def val_dataloader(self):
        return td.DataLoader(self.val_dataset, batch_size=2048, num_workers=0)

    def test_dataloader(self):
        return td.DataLoader(self.test_dataset, batch_size=512, shuffle=False, num_workers=0)  
    
class NNWrapper():
    def __init__(self, model_name:str, folder_name:str, df_train: pd.DataFrame, n_recommendations:int = 10, n_users:int = 24983, n_items = 100):
        self.n_users = n_users
        self.n_items = n_items
        self.model_name = model_name
    
        embed = 32
        self.model = ContextualRanker.load_from_checkpoint(os.path.join(folder_name, model_name),
        map_location=torch.device("cpu")
    , embedding_dim=embed, n_users=n_users, n_items=n_items, ratings_range=[-10, 10])

        self.n_recommendations = n_recommendations
        self.df_train = df_train.copy()
        
        
        
        self.helper = Helper()

    def predict(self) -> List[list]:
        t1 = time.time()
        test_set = self.helper.create_test_set(self.n_users, self.n_items)                
        df_test = pd.DataFrame(test_set, columns=['UID', 'JID', 'Rating'])
        
        df_test['UID'] = df_test['UID'].astype(int)
        df_test['JID'] = df_test['JID'].astype(int)

        data_module_test = ContextualRankerData(df_test, df_test, df_test, features = ["UID", "JID"])
        data_module_test.prepare_data()
        data_module_test.setup()

        predictions = []
        for x, y in data_module_test.test_dataloader():
            predict = self.model(x)
            predictions.extend(predict.cpu().detach().numpy())
    
    
        df_test['Rating_pred'] = predictions
        
        res = self.helper.filter_viewed_items(self.df_train, df_test, self.n_users, self.n_recommendations)
        print(self.model_name, 'time:', round(time.time() - t1, 3))
        
        #res = 
        return res
    