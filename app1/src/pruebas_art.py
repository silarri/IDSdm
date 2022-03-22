from data_handler import DATA_HANDLER
from data_miners import *
from matplotlib import pyplot
import pandas as pd


data_handler = DATA_HANDLER("../data/dataset1")
data_handler.load_data(verbose=True)

#aux = LOGREG(0,0)
aux = DTREE(0,0)  
aux.train_model(*data_handler.get_train_data())
cols = data_handler.get_train_data()[0].columns

print (cols)

#importance = aux.model.coef_
importance = aux.model.feature_importances_

# plot feature importance
#print (sorted(zip(importance, range(len(importance)))))
for val,idx in reversed(sorted(zip(importance, range(len(importance))))):
	print('Feature: %s, Score: %.5f' % (cols[idx],val))

#[imp, index for imp, index in sorted(zip(importance, range(len(importance))))]

pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
