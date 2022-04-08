import pandas as pd
from data_miners import *
from matplotlib import pyplot

df_train = pd.read_csv("../data/train/SSH_FTP_ISCX.csv") 

y_train = df_train.iloc[:,-1].values #Extract labels: 1 => normal, 0 => ANOMALY
x_train = df_train.iloc[: , :-1]     #Remove labels column

#normalization 
x_train  = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train)).values
#drop NAN columns
x_train.dropna( axis = 1, inplace=True)
cols = x_train.columns

model = DTREE(0,0)  
metrics = model.K_fold_cross_val(x_train,y_train,10,True)

for key, value in metrics.items():
        value = np.asarray(value)
        tab = ":\t"
        if key=="Recall" or key =="Fscore": tab = ":\t\t"
        print(key + tab + str(value.mean()))


importance = model.model.feature_importances_

# plot feature importance
#print (sorted(zip(importance, range(len(importance)))))
new_cols = []
for val,idx in reversed(sorted(zip(importance, range(len(importance))))):
	#print('Feature: %s, Score: %.5f' % (cols[idx],val))
    if val > 0:
        new_cols.append(cols[idx])

#print(new_cols)

x_train2 = x_train[new_cols]
model = DTREE(0,0)  
metrics = model.K_fold_cross_val(x_train2,y_train,10,True)

for key, value in metrics.items():
        value = np.asarray(value)
        tab = ":\t"
        if key=="Recall" or key =="Fscore": tab = ":\t\t"
        print(key + tab + str(value.mean()))



#pyplot.bar([x for x in range(len(importance))], importance)
#pyplot.show()



