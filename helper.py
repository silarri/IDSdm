#Library to load network data
import os
import numpy as np
import pandas as pd


#######main###########

df_train = pd.read_csv("./project/data/dataset2/train/train_data.csv")
df_train.loc[df_train["class"] != "normal", "class"] = "anomaly"
df_train.to_csv("./project/data/dataset2/train/train_data2.csv", index=False) 