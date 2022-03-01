from data_handler import DATA_HANDLER
from data_miners import LOGREG
from data_miners import KNN


####################################### LOAD AND PREPROCESS THE DATASET #####################

data_handler = DATA_HANDLER("../data")
data_handler.load_data(verbose=True)

######################################  TRAIN AND TEST DIFFERENT CLASSIFIERS WITH cross-val ###############

print("CHECKING with KFOLD CROSS VALIDATION (k=10) the expected capabilities of different learners")

print("LOGISTIC REGRESSION:",end='\t',flush=True)
logreg=LOGREG()
mean,std=logreg.cross_val(*data_handler.get_train_data())
print("%0.8f accuracy with a standard deviation of %0.8f" % (mean,std))

print("KNN:",end='\t\t\t',flush=True)
knn=KNN()
mean,std=knn.cross_val(*data_handler.get_train_data())
print("%0.8f accuracy with a standard deviation of %0.8f" % (mean,std))


#Apply K-fold-validation to see what scores we can expect
#Work with thresholds of confidence
#Create a multi-model system that can detect intrussions combining results and traininv with all the data
#Translate test data to train data after running it through the final system

