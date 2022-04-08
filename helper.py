#Library to load network data
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#######main###########

#df_train = pd.read_csv("/home/arturo/Uni/4º/TFG/Tuesday-WorkingHours.pcap_ISCX.csv")
#df_train.loc[df_train["class"] != "BENIGN", "class"] = "1" #1 MEANS INTRUSSION
#df_train.loc[df_train["class"] == "BENIGN", "class"] = "0" #0 MEANS NORMAL
#df_train.to_csv("/home/arturo/Uni/4º/TFG/SSH_FTP_ISCX.csv", index=False) 

df_train = pd.read_csv("/home/arturo/Uni/4º/TFG/TFG/app2/data/train/SSH_FTP_ISCX.csv")
y = df_train.iloc[:,-1].values #Extract labels: 1 => normal, 0 => ANOMALY
X = df_train.iloc[: , :-1]     #Remove labels column

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train = pd.DataFrame(X_train, columns = df_train.columns[:-1])
X_test = pd.DataFrame(X_test, columns = df_train.columns[:-1])
y_train = pd.DataFrame(y_train, columns = ['class'])
y_test = pd.DataFrame(y_test, columns = ['class'])

X_one = pd.concat([X_train,y_train],axis=1)
print(X_one)
