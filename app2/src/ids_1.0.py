import os
import time
import subprocess
import signal
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

########### GLOBAL CONTROL VARIABLES ###########

SNIFFING_WINDOW = 45 #Seconds
INTERFACE = "enp1s0" #interface used for sniffing
OUTPUT_FILE = "/tmp/flows.csv"  #File to dump the sniffed network traffic
CMD = "sudo cicflowmeter -i " + INTERFACE + " -c " + OUTPUT_FILE
TRAIN_DATA_FILE = "./SSH_FTP_ISCX.csv"
CONFIDENCE_THRESHOLD = 0.9 # 90%
FOUND_INTRUSIONS_FILE = "/home/ids_intrusions.csv"

########## DATA HANDLING CLASS ############

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

########## DATA MINER CLASS ############

class DATA_MINER():

    def __init__(self,pca_rfe=0,n_features=10):
        self.model=None
    
    #trains the model with all the avaiable data
    def train_model(self,X,y): 
        self.model.fit(X,y)

    #predicts the probability that a network transmission IS an intrusion
    def predict_proba_intrusion(self,x):
        aux = self.model.predict_proba(x) #returns firs probability of intrusion and then normal in a row
        aux = np.asarray(aux)
        prob_intrusion = aux.transpose()[0]
        return prob_intrusion

class DTREE(DATA_MINER):

    def __init__(self):
        super().__init__()
        self.model=DecisionTreeClassifier()

#On startup train the prediction model with all the data

data_handler = DATA_HANDLER(TRAIN_DATA_FILE)
if not data_handler.load_data():
    exit(1)

model = DTREE()
model.train_model(*data_handler.return_data()) #PERFORM REDUCTION?? No need really

while(1):

#Perform sniffing:

    with open(os.devnull, 'w') as fp:
       proc = subprocess.Popen(['sudo','cicflowmeter','-i',INTERFACE,'-c',OUTPUT_FILE],stdout=fp,preexec_fn=os.setsid)

    print("SINIFFING")
    time.sleep(SNIFFING_WINDOW) # <-- There's no time.wait, but time.sleep.
    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    proc.wait() #Wait for sniffing to stop in an orderly fashion
    print("SINIFFING Done")

#Read sniffed data and predict
    if not os.path.exists(OUTPUT_FILE) or os.stat(OUTPUT_FILE).st_size == 0: #file doesnt exist or is empty
        print("empty file")
        continue
    original = pd.read_csv(OUTPUT_FILE)
    sniffed_data = original[data_handler.column_names] #Select the desired columns
    probs = model.predict_proba_intrusion(sniffed_data)
    predictions = [True if y >= CONFIDENCE_THRESHOLD else False for y in probs] 
    print(predictions)

    #Intrusion?
    if True in predictions: #Detected an intrusion: Save to file and email root
        original["confidence"]=probs #Add confidence of prediction
        original[predictions].to_csv(FOUND_INTRUSIONS_FILE, mode='a', header=not os.path.exists(FOUND_INTRUSIONS_FILE))
        N_intrusions=str(len(original[predictions]))
        cmd = "echo \""+N_intrusions+" possible malicious connections detected.\nStored in the file "+FOUND_INTRUSIONS_FILE+".Please take the appropiate actions.\" | mail -s \"IDS ALERT\" root"
        os.system(cmd)
    #Else keep on going!


