import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from data_handler import DATA_HANDLER
from data_miners import LOGREG,KNN,DTREE,GNB,MLPC
from math import sqrt

#Metrics that will be used across the TestBench: 
#Accuracy(Exactitud):   The fraction of predictions our model got right.
#Precision:             The number of TP divided by the total number of positive predictions. 
#Recall:                The ability of a model to find all the cases of a class within a data set.
#F1score:               The harmonic mean between precision and recall. The higher the better.
#ConfMatrix             Grafically shows the correct and incorrect predictions


########## Welcome and error functions #########################################
def print_welcome():
    print ("Welcome to the TESTBENCH for IDS-NET \n")
    print ("Please indicate the relative path to the folder containing the training and testing data.")
    print ("It MUST follow the following structure:\n")
    print ("folder/")
    print ("|-train/")
    print ("|   |-train_data.csv")
    print ("|-test/")
    print ("|   |-test_data.csv")
    path = input(":")
    return path

def print_exit(error=0):
    if error == 0:
        print("\nBYE!")
        exit(0)
    elif error == 1:
        print("\nABORTING EXECUTION! Encountered a problem with the data directory!")
    elif error == 2:
        print("\nABORTING EXECUTION! Two or more algorithms are needed!")
    elif error == 3:
        print("\nABORTING EXECUTION! Invalid tesbench option!")
    elif error == 4:
        print("\nABORTING EXECUTION! Number of features must be an integer")
    elif error == 5:
        print("\nABORTING EXECUTION! Unknown algorithm")
    elif error == 6:
        print("\nABORTING EXECUTION! Please respect the range syntax")

    exit(1)

########## Functions to ask the user for inputs ########################################

def select_tb_option():
    print("Choose one of the following options:")
    print("Option 1: Compare algorithms amongst themselves with the same settings")
    print("Option 2: Compare the performance of one algorithm with different settings")
    print("Option 3: Perform hyperparameter tuning")
    option = input("Please introduce the desired option (1,2,3):")

    if option not in ["1","2","3"]:
        print_exit(error=3) 
    
    option = int(option)

    file_name = "one_vs_one_"
    if option == 1:
        file_name = "all_vs_all_"

    return option, file_name

def select_algos(option,file_name):
    avaiable_models = ["LOGREG","KNN","DTREE","GNB","MLPC"]
    if option == 1: #All vs All, list of at least 2 algorithms
        algos= input("Insert a list of the algorithms to compare (LOGREG,KNN,DTREE,GNB,MLPC): ")
        algos = algos.split()  
        for i in algos:
            if i not in avaiable_models:
                algos.remove(i)
            else:
                file_name = file_name + i + "_"

        if len(algos) < 2:
            print_exit(error=2)

    elif option == 2: #option 2

        algos= input("Insert ONE algorithm (LOGREG,KNN,DTREE,GNB,MLPC): ")

        if algos not in avaiable_models:
            print_exit(error=5)

        file_name = file_name + algos + "_"
    else: #option 3

        algos= input("Insert ONE algorithm (KNN): ")

        if algos not in avaiable_models:
            print_exit(error=5)

        file_name = ""

    return algos, file_name

def select_pca_rfe(file_name):
    pca_rfe=input("Select an option for feature selection: (PCA,RFE,NONE)\n(RFE is not avaiable with KNN, GNB & MLPC):")
    if pca_rfe == "PCA":
        file_name = file_name + "PCA_"
        pca_rfe=1
    elif pca_rfe=="RFE":
        file_name = file_name + "RFE_"
        pca_rfe=2
    else: #NONE
        pca_rfe=0

    return pca_rfe, file_name

def select_n_features(pca_rfe,file_name):
    n_features=0
    if pca_rfe == 1 or pca_rfe == 2:
        n_features = input("Insert the number of features to select (int):")
        try:
            n_features = int(n_features)
        except ValueError:
            print_exit(error=4)

        file_name = file_name + str(n_features)
    
    return n_features, file_name

def select_k_folds(file_name):
    folds = input("Insert the number of splits to perform K-fold (int >=2):")
    try:
        folds = int(folds)
    except ValueError:
        folds = 10 #Default value
    if folds < 2: folds = 2

    file_name = file_name + "k" +str(folds)
    return folds, file_name

def select_k_range():
    range_ = input("Insert a range of values to try for K with this syntax (int): start stop step\n")
    range_ = range_.split()
    if len(range_) != 3: 
        print_exit(error=5)
    try:
        range_ = [int(i) for i in range_]
    except ValueError:
        print_exit(error=5)

    return range_


########## Finctions for all_vs_all testbenching #########################################
def print_results(metrics_dict,pretty_name):
    print("\nAverage metrics for "+ pretty_name+":")

    for key, value in metrics_dict.items():
        value = np.asarray(value)
        tab = ":\t"
        if key=="Recall" or key =="Fscore": tab = ":\t\t"
        print(key + tab + str(value.mean()))

def all_vs_all_tb(data_handler,algos,pca_rfe,n_features,file_name,splits,verbose=True):
    constructors=[]
    for algo in algos:
        if algo == "LOGREG":
            aux = LOGREG(pca_rfe,n_features)
        if algo == "KNN":
            k_neigh = input("Please insert the number of neighbors to use in KNN")
            k_neigh = int(k_neigh)
            aux = KNN(n_neighbors_=k_neigh, pca_rfe=pca_rfe,n_features=n_features)  
        if algo == "DTREE":
            aux = DTREE(pca_rfe,n_features)      
        if algo == "GNB":
            aux = GNB(pca_rfe,n_features)    
        if algo == "MLPC":
            aux = MLPC(pca_rfe,n_features)          
        constructors.append(aux)

    all_metrics = []
    for model in constructors:
        metrics = model.K_fold_cross_val(*data_handler.get_train_data(),splits,True)
        all_metrics.append((metrics,model.pretty_name))
    
    if verbose: 
        for metric, name in all_metrics:
            print_results(metric,name)

    #output results to csv file
    header = ["Algorithm","Accuracy","Precision","Recall","Fscore"]
    data = []

    for metrics, name in all_metrics:
        row = [name]
        for _, value in metrics.items():
            value = np.asarray(value)
            m = value.mean()
            row.append( "{0:.4f}".format(m*100))
        data.append(row)

    with open(file_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def one_vs_one_tb(data_handler,algo,n_features,file_name,splits,verbose=True):
    constructors = []
    if algo == "LOGREG":
        constructors.append(LOGREG(0,n_features))
        constructors.append(LOGREG(1,n_features))
        constructors.append(LOGREG(2,n_features))
    if algo == "KNN":
        k_neigh = input("Please insert the number of neighbors to use in KNN")
        k_neigh = int(k_neigh)
        constructors.append(KNN(n_neighbors_=k_neigh,pca_rfe=0,n_features=n_features))
        constructors.append(KNN(n_neighbors_=k_neigh,pca_rfe=1,n_features=n_features))
    if algo == "DTREE":
        constructors.append(DTREE(0,n_features))
        constructors.append(DTREE(1,n_features))
        constructors.append(DTREE(2,n_features))     
    if algo == "GNB":
        constructors.append(GNB(0,n_features))
        constructors.append(GNB(1,n_features))
    if algo == "MLPC":
        constructors.append(MLPC(0,n_features))
        constructors.append(MLPC(1,n_features))

    all_metrics = []
    for model in constructors:
        metrics = model.K_fold_cross_val(*data_handler.get_train_data(),splits,True)
        all_metrics.append((metrics,model.full_name))
    
    if verbose: 
        for metric, name in all_metrics:
            print_results(metric,name)

    #output results to csv file
    header = ["Algorithm","Accuracy","Precision","Recall","Fscore"]
    data = []

    for metrics, name in all_metrics:
        row = [name]
        for _, value in metrics.items():
            value = np.asarray(value)
            m = value.mean()
            row.append( "{0:.4f}".format(m*100))
        data.append(row)

    with open(file_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def hyperparameter_tuning(data_handler,algo):
    if algo == "KNN":
        folds,_ = select_k_folds("")
        range_  = select_k_range()
        ks,scores = KNN().tune_hyperparameter_k(*data_handler.get_train_data(),range_[0],range_[1],range_[2],splits=folds,verbose=True)
        #plt.figure(figsize=(12,6))
        plt.figure()
        plt.plot(ks,scores, marker='o', color='r',)
        plt.ylabel('accuracy score')
        plt.xlabel('K neighbors')
        plt.grid('on')
        plt.title('KNN k tuning')
        plt.show()
        
#############################################  MAIN   ########################################

#print welcome message
path = print_welcome()

#Load the provided data
data_handler = DATA_HANDLER(path)
if not data_handler.load_data(verbose=True):
    print_exit(error=1)
    exit(1)

#ASK the user to select a type of Testbench: (All vs All or Different settings for one Algo)
option, file_name=select_tb_option()

#ASK the user to select the Algorithms to be tested (either a list or one algorithm)
algos, file_name = select_algos(option,file_name)

if option == 1: #Ask the user for the common parameters
    #Ask the user if they want to use PCA or RFE
    pca_rfe, file_name = select_pca_rfe(file_name)

    #Ask the user for the number of features to select
    n_features, file_name = select_n_features(pca_rfe,file_name)

    #Ask the user for the number of folds to perform:
    kfolds, file_name = select_k_folds(file_name)

    #Execute All_vs_All testbench
    file_name = file_name + "_TB.csv"
    file_name = os.path.join(path,file_name)
    all_vs_all_tb(data_handler,algos,pca_rfe,n_features,file_name,kfolds,verbose=True)
    print("The file "+ os.path.join(path,file_name)+" has been created.")

elif option == 2: #option 2 one vs one
    #Ask the user for the number of features to select
    n_features, _ = select_n_features(1,file_name)

    #Ask the user for the number of folds to perform:
    kfolds, file_name = select_k_folds(file_name)

    #Execute ONE vs ONE testbench
    file_name = file_name + "_TB.csv"
    file_name = os.path.join(path,file_name)
    one_vs_one_tb(data_handler,algos,n_features,file_name,kfolds,verbose=True)
    print("The file "+ os.path.join(path,file_name)+" has been created.")

else: #option 3
    #Perform hyperparameter tuning
    hyperparameter_tuning(data_handler,algos)

print_exit()


#TODO: Pasa pq algún split del Kfold no tiene ambas clases (normal y anomaly)
#Solucionado haciendo shuffle pero no está asegurado!

#Logistic Regression :==/home/arturo/.local/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
#  _warn_prf(average, modifier, msg_start, len(result))


######################################## LOAD AND PREPROCESS THE DATASET #####################
#
#pca_rfe=input("Do you want to perfom PCA or RFE to the DATA?(PCA,RFE,NONE) (RFE won't be performed with KNN & #GNB):")
#if pca_rfe == "PCA":
#    pca_rfe=1
#elif pca_rfe=="RFE":
#    pca_rfe=2
#else: #NONE
#    pca_rfe=0
#
#n_features = 0 #Default value, will use all features
#if pca_rfe == 1 or pca_rfe == 2:
#    n_features = input("Insert the number of features to be used (int):")
#    n_features = int(n_features)
#
#
#data_handler = DATA_HANDLER("../data")
#data_handler.load_data(verbose=True)
#
#######################################  TRAIN AND TEST DIFFERENT CLASSIFIERS WITH cross-val ###############
#
#print("\nCHECKING with KFOLD CROSS VALIDATION (k=10) the expected capabilities of different learners")
#
##print("LOGISTIC REGRESSION:",end='\t',flush=True)
##logreg=LOGREG(pca_rfe,n_features)
##mean,std=logreg.cross_val(*data_handler.get_train_data())
##print("%0.8f accuracy with a standard deviation of %0.8f" % (mean,std))
##
##print("KNN:",end='\t\t\t',flush=True)
##knn=KNN(pca_rfe=pca_rfe,n_features=n_features)
##mean,std=knn.cross_val(*data_handler.get_train_data())
##print("%0.8f accuracy with a standard deviation of %0.8f" % (mean,std))
##
##print("DTREE:",end='\t\t\t',flush=True)
##dtree=DTREE(pca_rfe,n_features)
##mean,std=dtree.cross_val(*data_handler.get_train_data())
##print("%0.8f accuracy with a standard deviation of %0.8f" % (mean,std))
##
##print("GNB:",end='\t\t\t',flush=True)
##gnb=GNB(pca_rfe,n_features)
##mean,std=gnb.cross_val(*data_handler.get_train_data())
##print("%0.8f accuracy with a standard deviation of %0.8f" % (mean,std))
##
#print("MLPC:",end='\t\t\t',flush=True)
#mplc=MLPC(pca_rfe,n_features)
#mean,std=mplc.cross_val(*data_handler.get_train_data())
#print("%0.8f accuracy with a standard deviation of %0.8f" % (mean,std))
#
##Apply K-fold-validation to see what scores we can expect
##Work with thresholds of confidence
##Create a multi-model system that can detect intrussions combining results and traininv with all the data
##Translate test data to train data after running it through the final system
#
#