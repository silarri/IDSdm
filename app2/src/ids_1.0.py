"""
Author: Arturo Calvera Tonin
Date: June 2022
Project: TFG - Data mining for intrusion detection in communication networks
File: ids_1.0.py
Comms: Python file with the code for a linux daemon style intrusion detection system.
Read the Readme.txt with the instructions to setup the daemon and the dependencies of the system
"""
import os
import sys
import time
import subprocess
import signal
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support

########### GLOBAL CONTROL VARIABLES ###########

INTERFACE = "enp1s0" #interface used for sniffing
OUTPUT_FILE = "/tmp/flows.csv"  #File to dump the sniffed network traffic
SIM_INPUT_FILE = "/etc/ids/SSH_FTP_ISCX_test.csv" #File used to debug the service bypassing network sniffing
CMD = "sudo cicflowmeter -i " + INTERFACE + " -c " + OUTPUT_FILE
TRAIN_DATA_FILE = "/etc/ids/SSH_FTP_ISCX_train.csv"
CONFIDENCE_THRESHOLD = 0.9 # 90%
FOUND_INTRUSIONS_FILE = "/etc/ids/ids_intrusions.csv"
MODE = 1 if len(sys.argv)<=1 else 0 #Default working mode is SERVICE. 0 == DEBUG MODE


########## REDUCED DATA HANDLING CLASS ############

class DATA_HANDLER:

    def __init__(self,data_file):
        self.data_file=data_file            #Directory with the data-set to be used

    def load_data(self):
        df_train = pd.read_csv(self.data_file)

        #We assumme the class is the last column (binary)
        y_train = df_train.iloc[:,-1].values #Extract labels: 1 => normal, 0 => ANOMALY
        x_train = df_train.iloc[: , :-1]     #Remove labels column
        
        #normalization (min-max scaling) 
        x_train  = (x_train-x_train.min())/(x_train.max()-x_train.min())

        #drop NAN columns
        x_train.dropna( axis = 1, inplace=True)

        self.X = x_train
        self.y = y_train
        self.column_names= self.X.columns

        return True

    def return_data(self):
        return self.X, self.y

########## REDUCED DATA MINER CLASS ############

class DATA_MINER():

    def __init__(self):
        self.model=None
    
    #trains the model with all the avaiable data
    def train_model(self,X,y): 
        self.model = self.model.fit(X,y)

    #predicts the probability that a network transmission IS an intrusion 1=>normal 0=> intrusion
    def predict_proba_intrusion(self,x):
        aux = self.model.predict_proba(x) #returns first probability of intrusion and then normal in a row
        aux = np.asarray(aux)
        prob_intrusion = aux.transpose()[0]
        return prob_intrusion

class DTREE(DATA_MINER):

    def __init__(self):
        super().__init__()
        self.model=DecisionTreeClassifier()


############################## MAIN #########################################

#On startup train the prediction model with all the data
data_handler = DATA_HANDLER(TRAIN_DATA_FILE)
if not data_handler.load_data():
    exit(1)

classifier = DTREE()
classifier.train_model(*data_handler.return_data()) #PERFORM REDUCTION?? No need really

if MODE == 0: #DEBUG MODE:

    sim_input = pd.read_csv(SIM_INPUT_FILE)
    y_real = sim_input.iloc[:,-1].values
    sim_input = sim_input[data_handler.column_names]
    #normalization (min-max scaling) 
    sim_input  = (sim_input-sim_input.min())/(sim_input.max()-sim_input.min())
    #drop NAN columns
    sim_input.dropna( axis = 1, inplace=True)

    probs = classifier.predict_proba_intrusion(sim_input)
    predictions = [0 if y >= CONFIDENCE_THRESHOLD else 1 for y in probs] 
    p, r, f, _ = precision_recall_fscore_support(y_real, predictions, average='weighted') 
    score = classifier.model.score(sim_input,y_real)
    
    print("Accuracy: ", score)
    print("Precision: ", p)
    print("Recall: ", r)
    print("Fscore: ", f)
    exit(0)
   

while(1):

#Perform network analysis:
    print("Analyzing network traffic for 60 seconds")
    with open(os.devnull, 'w') as fp:
        proc = subprocess.Popen(['sudo','cicflowmeter','-i',INTERFACE,'-c',OUTPUT_FILE],stdout=fp,start_new_session=True)
        #print("Waiting")
        proc.wait()  #Cicflow is supposed to stop after 60 seconds

#Read sniffed data and predict
    if not os.path.exists(OUTPUT_FILE) or os.stat(OUTPUT_FILE).st_size == 0: #file doesnt exist or is empty
        #print("empty file")
        continue
    original = pd.read_csv(OUTPUT_FILE)
    sniffed_data = original[data_handler.column_names] #Select the desired columns
    probs = model.predict_proba_intrusion(sniffed_data)
    predictions = [True if y >= CONFIDENCE_THRESHOLD else False for y in probs] 
    #print(predictions)

    #Intrusion?
    if True in predictions: #Detected an intrusion: Save to file and email root
        original["confidence"]=probs #Add confidence of prediction
        original[predictions].to_csv(FOUND_INTRUSIONS_FILE, mode='a',index=False, header=not os.path.exists(FOUND_INTRUSIONS_FILE))
        N_intrusions=str(len(original[predictions]))
        cmd = "echo \""+N_intrusions+" possible malicious connections detected.\nStored in the file "+FOUND_INTRUSIONS_FILE+".Please take the appropiate actions.\" | mail -s \"IDS ALERT\" root"
        os.system(cmd)
    #Else keep on going!


