import pandas as pd
from data_miners import *

df_train = pd.read_csv("../data/train/SSH_FTP_ISCX.csv") 

y_train = df_train.iloc[:,-1].values #Extract labels: 1 => normal, 0 => ANOMALY
x_train = df_train.iloc[: , :-1]     #Remove labels column

#normalization 
x_train  = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train)).values
#drop NAN columns
x_train.dropna( axis = 1, inplace=True)

model = DTREE(0,0)  
metrics = model.K_fold_cross_val(x_train,y_train,10,True)

print(metrics)



