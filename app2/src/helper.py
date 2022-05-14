#Library to load network data
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#######main###########

#df_train = pd.read_csv("/home/arturo/Uni/4º/TFG/SSH_FTP_ISCX.csv")
#df_train.loc[df_train["class"] != "BENIGN", "class"] = "0" #0 MEANS INTRUSSION
#df_train.loc[df_train["class"] == "BENIGN", "class"] = "1" #1 MEANS NORMAL
#df_train.to_csv("/home/arturo/Uni/4º/TFG/SSH_FTP_ISCX.csv", index=False) 

#df_train = pd.read_csv("/home/arturo/Uni/4º/TFG/TFG/app2/data/train/SSH_FTP_ISCX.csv")
#df_train.loc[df_train["class"] == 1, "class"] = "0" #0 MEANS INTRUSSION
#df_train.loc[df_train["class"] == 0, "class"] = "1" #1 MEANS NORMAL
#df_train.to_csv("/home/arturo/Uni/4º/TFG/TFG/app2/data/train/SSH_FTP_ISCX.csv", index=False) 

df_train = pd.read_csv("/home/arturo/Uni/4º/TFG/TFG/app2/data/train/SSH_FTP_ISCX_train.csv")
#df_train.loc[df_train["class"] == 1, "class"] = "norma" #0 MEANS NORMAL, NEGATIVE
#df_train.loc[df_train["class"] == 0, "class"] = "anomaly" #1 MEANS INTRUSION, POSITIVE
df_train.loc[df_train["class"] == "norma", "class"] = "0" #0 MEANS NORMAL, NEGATIVE
df_train.loc[df_train["class"] == "anomaly", "class"] = "1" #1 MEANS INTRUSION, POSITIVE
df_train.to_csv("/home/arturo/Uni/4º/TFG/TFG/app2/data/train/SSH_FTP_ISCX_train.csv", index=False)

#y = df_train.iloc[:,-1].values #Extract labels: 1 => normal, 0 => ANOMALY
#X = df_train.iloc[: , :-1]     #Remove labels column
#
#normal = [i for i in y if i == 1]
#anormal = [i for i in y if i == 0]
#print(len(normal),len(anormal))
#train, test = train_test_split(df_train, test_size=0.2)
#train.to_csv("/home/arturo/Uni/4º/TFG/TFG/app1/data/dataset4/train/SSH_FTP_ISCX_train.csv", index=False) 
#test=test.iloc[: , :-1] #PUEDO NO ELIMINARLAS Y USARLAS COMO SOLUCION!!!
#test.to_csv("/home/arturo/Uni/4º/TFG/TFG/app1/data/dataset4/test/SSH_FTP_ISCX_test.csv", index=False) 

#details = test.apply(lambda x : True if x['class'] == 0 else False, axis = 1)
#
#print ("TEST", len(test),len(details[details == True].index))
#details = train.apply(lambda x : True if x['class'] == 0 else False, axis = 1)
#print ("TRAIN", len(train),len(details[details == True].index))
#X_train = pd.DataFrame(X_train, columns = df_train.columns[:-1])
#X_test = pd.DataFrame(X_test, columns = df_train.columns[:-1])
#y_train = pd.DataFrame(y_train, columns = ['class'])
#y_test = pd.DataFrame(y_test, columns = ['class'])
#
#X_one = pd.concat([X_train,y_train],axis=1)
#print(X_one)
