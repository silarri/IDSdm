#Library to load network data
import os
import numpy as np
import pandas as pd

#Comms: Class with useful functions to load and process the network data in .csv format
#PRE: The directory "data_directory" should have two subdectories "train" and "test" with the training and testing data grouped in a single .csv file respectively. The testing data is NOT Labeled
#POST:

#TODO: Debería aceptar subdirectorios con más de un CSV?? Qué pasa con distintos formatos de datos??

class DATA_HANDLER:

    def __init__(self,data_directory):
        self.data_dir=data_directory            #Directory with the data-set to be used

    def load_data(self):

        print ("LOADING DATASET...")
        #Check for correct data directory structure

        train_dir = os.path.join(self.data_dir,"train")
        test_dir  = os.path.join(self.data_dir,"test")

        if not self.__check_dir(train_dir,test_dir): return False
        
        df_train = pd.read_csv(str(train_dir)+"/"+os.listdir(train_dir)[0]) 
        df_test = pd.read_csv(str(test_dir)+"/"+os.listdir(test_dir)[0]) #Not labeled

        self.x_train, self.y_train, self.x_unlabeled = self.__preprocess_data(df_train,df_test)

        print ("LOADING DONE: " + str(len(self.x_train))+ " records avaiable for training")
        
        return True

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

        x_test = df_test

        #normalization 
        x  = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train)).values
        x_ = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test)).values
        #drop NAN columns
        x.dropna( axis = 1, inplace=True)
        x_.dropna( axis = 1, inplace=True)
        
        return x,y_train,x_


    #Checks for correct structure in the data directory 
    def __check_dir(self,train_dir,test_dir):
        
        if not os.path.exists(train_dir):
            print ("Unable to find training data, aborting.")
            return False
        elif len(os.listdir(train_dir)) != 1 or not os.listdir(train_dir)[0].endswith('.csv'):
            print ("The training directory should contain one and only one \'.csv\' file.")
            return False

        if not os.path.exists(test_dir):
            print ("Unable to find testing data, aborting.")
            return False
        elif len(os.listdir(test_dir)) != 1 or not os.listdir(test_dir)[0].endswith('.csv'):
            print ("The testing directory should contain one and only one \'.csv\' file.")
            return False
        
        return True
