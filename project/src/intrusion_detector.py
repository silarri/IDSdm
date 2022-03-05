import numpy as np
import os
from data_handler import DATA_HANDLER
from data_miners import LOGREG,KNN,DTREE,GNB

class intrusion_detector():

    def __init__(self,data_handler,miners,threshold,pca_rfe,n_features):
        self.threshold=threshold        #Confidence threshold
        self.miners = []                #will hold the trained models
        self.data_handler=data_handler
        self.pca_rfe=pca_rfe
        avaiable_models=["LOGREG","KNN","DTREE","GNB"]

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
            aux.train_model(*data_handler.get_train_data())
            self.miners.append(aux)

        if not self.miners: #Is empty
            #None of the algorithms introduced is avaiable
            raise RuntimeError("None of the algorithms introduced is avaiable")

            #TODO: Add more algos

    def find_intrusions(self,folder): #Checks weather a network exchange is normal or fraudulent

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
        probs = probs.mean(axis=0) #Compute the mean prob of intrusion using all models

        probs = adjust_all_predictions(probs)

        self.data_handler.original_test.iloc[probs].to_csv(os.path.join(folder,"intrusions.csv"), index=False) 

        


