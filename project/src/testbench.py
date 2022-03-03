from data_handler import DATA_HANDLER
from data_miners import LOGREG,KNN,SVC_,DTREE


####################################### LOAD AND PREPROCESS THE DATASET #####################

pca_rfe=input("Do you want to perfom PCA or RFE to the DATA?(PCA,RFE,NONE) (RFE won't be performed with KNN):")
if pca_rfe == "PCA":
    pca_rfe=1
elif pca_rfe=="RFE":
    pca_rfe=2
else: #NONE
    pca_rfe=0

if pca_rfe == 1 or pca_rfe == 2:
    n_features = input("Insert the number of features to be used (int):")
    n_features = int(n_features)


data_handler = DATA_HANDLER("../data")
data_handler.load_data(verbose=True)

######################################  TRAIN AND TEST DIFFERENT CLASSIFIERS WITH cross-val ###############

print("\nCHECKING with KFOLD CROSS VALIDATION (k=10) the expected capabilities of different learners")

print("LOGISTIC REGRESSION:",end='\t',flush=True)
logreg=LOGREG(pca_rfe,n_features)
mean,std=logreg.cross_val(*data_handler.get_train_data())
print("%0.8f accuracy with a standard deviation of %0.8f" % (mean,std))

print("KNN:",end='\t\t\t',flush=True)
knn=KNN(pca_rfe=pca_rfe,n_features=n_features)
mean,std=knn.cross_val(*data_handler.get_train_data())
print("%0.8f accuracy with a standard deviation of %0.8f" % (mean,std))

#print("SVC:",end='\t\t\t',flush=True)
#SVC=SVC_()
#mean,std=SVC.cross_val(*data_handler.get_train_data())
#print("%0.8f accuracy with a standard deviation of %0.8f" % (mean,std))

print("DTREE:",end='\t\t\t',flush=True)
dtree=DTREE(pca_rfe,n_features)
mean,std=dtree.cross_val(*data_handler.get_train_data())
print("%0.8f accuracy with a standard deviation of %0.8f" % (mean,std))

#Apply K-fold-validation to see what scores we can expect
#Work with thresholds of confidence
#Create a multi-model system that can detect intrussions combining results and traininv with all the data
#Translate test data to train data after running it through the final system

