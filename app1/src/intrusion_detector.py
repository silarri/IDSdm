"""
Author: Arturo Calvera Tonin
Date: June 2022
Project: TFG - Data mining for intrusion detection in communication networks
File: intrusion_detector.py
Comms: Library with an intrusion detection class based on machine learning
"""
import numpy as np
import os
from data_handler import DATA_HANDLER
from data_miners import LOGREG,KNN,DTREE,GNB

#Comms: Class that performs predictions on data to detect abnormal network traffic
#Uses the data_handler.py and data_miners.py library
class intrusion_detector():

    #data_handler: Receives a DATA_HANDLER object with the data already loaded
    #miners: Receives a strings list with the algorithms to use in the predictions
    def __init__(self,data_handler,miners,threshold,pca_rfe,n_features):
        self.threshold=threshold        #Confidence threshold
        self.miners = []                #will hold the trained models
        self.data_handler=data_handler
        self.pca_rfe=pca_rfe
        avaiable_models=["LOGREG","KNN","DTREE","GNB","MLPC"]

        for algo in miners:
            if algo not in avaiable_models:
                continue
            if algo == "LOGREG":
                aux = LOGREG(pca_rfe,n_features)
            if algo == "KNN":
                aux = KNN(pca_rfe=pca_rfe,n_features=n_features)  
            if algo == "DTREE":
                aux = DTREE(pca_rfe,n_features)      
            if algo == "GNB":
                aux = GNB(pca_rfe,n_features)   
            if algo == "MLPC":
                aux = MLPC(pca_rfe,n_features)           
            aux.train_model(*data_handler.get_train_data())
            self.miners.append(aux)

        if not self.miners: #Is empty
            #None of the algorithms introduced is avaiable
            raise RuntimeError("None of the algorithms introduced is avaiable")

    #Checks weather a network exchange is normal or fraudulent
    #agregate: Boolean to agregate or not the predictions of the different algorithms
    def find_intrusions(self,folder,agregate): 

        def adjust_all_predictions(predicted_probs):
            return [True if y >= self.threshold else False for y in predicted_probs] 
            #True => it's intrusion False=> normal

        entry = self.data_handler.get_test_data()
        probs = []  #Becomes a matrix with the estimations for each model 
                    #Each models predictions in a row, in each position probability for entry k

        #IF PCA was performed, Xtest needs to be transformed accordingly
        for miner in self.miners:
            if miner.pca_rfe==1:
                aux=miner.transform_data_with_pca(entry)
                row = miner.predict_proba_intrusion(aux)
            else:
                row = miner.predict_proba_intrusion(entry)
            probs.append(row)

        probs=np.asarray(probs)
        if agregate:
            probs = probs.mean(axis=0) #Compute the mean prob of intrusion using all models
            p = adjust_all_predictions(probs)
            aux = self.data_handler.original_test
            aux["confidence"]=probs
            aux[p].to_csv(os.path.join(folder,"intrusions.csv"), index=False) 
        else:
            for prob,miner in zip(probs,self.miners):
                p = adjust_all_predictions(prob)
                aux = self.data_handler.original_test
                aux["confidence"]=prob
                aux[p].to_csv(os.path.join(folder,miner.pretty_name+"_intrusions.csv"), index=False) 

        


