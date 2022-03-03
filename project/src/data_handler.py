#Library to load network data
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Comms: Class with useful functions to load and process the network data in .csv format
#PRE: The directory "data_directory" should have two subdectories "train" and "test" with the training and testing data grouped in a single .csv file respectively. The testing data is NOT Labeled
#POST:

#TODO: Debería aceptar subdirectorios con más de un CSV?? Qué pasa con distintos formatos de datos??

class DATA_HANDLER:

    def __init__(self,data_directory):
        self.data_dir=data_directory            #Directory with the data-set to be used

    def load_data(self,verbose=True):

        if verbose: print ("\nLOADING DATASET...",end='',flush=True)
        #Check for correct data directory structure

        train_dir = os.path.join(self.data_dir,"train")
        test_dir  = os.path.join(self.data_dir,"test")

        if not self.__check_dir(train_dir,test_dir): return False
        
        df_train = pd.read_csv(str(train_dir)+"/"+os.listdir(train_dir)[0]) 
        df_test = pd.read_csv(str(test_dir)+"/"+os.listdir(test_dir)[0]) #Not labeled
        
        self.original_test=df_test      #will be used to return the data of the fraudulent transactions
        self.X, self.y , self.x_unlabeled = self.__preprocess_data(df_train,df_test)

        if verbose: 
            print ("DONE: ")
            print ( str(len(self.X))+ " records avaiable for training")
            print ( str(len(self.x_unlabeled))+ " records avaiable for testing")
            #print ("Using "+ str(self.X.shape[1])+" features for training and testing")
        #if verbose:
        #    print("X:", self.X.shape)
        #    print("y:", self.y.shape)
        
        return True

    def get_train_data(self):
        return self.X, self.y

    def get_test_data(self):
        return self.x_unlabeled
    

    #Preprocess data to make it suitable for data mining
    def __preprocess_data(self,df_train,df_test):

        #Label decoding: Change qualitative columns to quantitative columns
        #In this case One Hot Encoding is choosen so as not to misslead the data mining algorithms with the weight of numerical values.

        qualitative_columns=list(df_train.select_dtypes(include=['object']).columns)
        df_train = pd.get_dummies(df_train,columns = qualitative_columns , drop_first=True)

        #We assumme the class is the last column (binary)
        y_train = df_train.iloc[:,-1].values #Extract labels: 1 => normal, 0 => ANOMALY
        x_train = df_train.iloc[: , :-1]     #Remove labels column

        qualitative_columns=list(df_test.select_dtypes(include=['object']).columns)
        df_test = pd.get_dummies(df_test,columns = qualitative_columns , drop_first=True)

        #Unlabeled test data
        x_test = df_test
        
        #normalization 
        x_train  = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train)).values
        x_test = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test)).values

        #drop NAN columns
        x_train.dropna( axis = 1, inplace=True)
        x_test.dropna( axis = 1, inplace=True)

        #One hot Encoding can cause different number of features if the values in the data aren't the same.
        #We need to add dummy columns to have the same features in the SAME ORDER

        #TODO: find a better way... WE MAY NOT HAVE THE TEST DATA AVAIABLE

        features=x_train.columns.union(x_test.columns)
        x_train = x_train.reindex(columns=features, fill_value=0)
        x_test = x_test.reindex(columns=features, fill_value=0)

        #print(x_test.shape,x_train.shape)

        return x_train,y_train,x_test


    #Checks for correct structure in the data directory 
    def __check_dir(self,train_dir,test_dir):
        
        if not os.path.exists(train_dir):
            print ("\nUnable to find training data, aborting.")
            return False
        elif len(os.listdir(train_dir)) != 1 or not os.listdir(train_dir)[0].endswith('.csv'):
            print ("\nThe training directory should contain one and only one \'.csv\' file.")
            return False

        if not os.path.exists(test_dir):
            print ("\nUnable to find testing data, aborting.")
            return False
        elif len(os.listdir(test_dir)) != 1 or not os.listdir(test_dir)[0].endswith('.csv'):
            print ("\nThe testing directory should contain one and only one \'.csv\' file.")
            return False
        
        return True
