import numpy as np
from data_handler import DATA_HANDLER
from data_miners import LOGREG,KNN

class intrusion_detector():

    def __init__(self,data_handler,miners,threshold):
        self.threshold=threshold        #Confidence threshold
        self.miners = []                #will hold the trained models

        for algo in miners:
            if algo == "LOGREG":
                aux = LOGREG()
            if algo == "KNN":
                aux = KNN()
            aux.train_model(*data_handler.get_train_data())
            self.miners.append(aux)

            #TODO: Add more algos

    def find_intrusions(self,entry): #Checks weather a network exchange is normal or fraudulent

        def adjust_all_predictions(predicted_probs):
            return [True if y >= self.threshold else False for y in predicted_probs] 
            #True => it's intrusion False=> normal

        probs = []  #Becomes a matrix with the estimations for each model 
                    #Each models predictions in a row, in each position probability for entry k
        for miner in self.miners:
            row = miner.predict_proba_intrusion(entry)
            probs.append(row)

        probs=np.asarray(probs)
        probs = probs.mean(axis=0) #Compute the mean prob of intrusion using all models

        probs = adjust_all_predictions(probs)

        compression_opts = dict(method='zip', archive_name='intrusions.csv')  
        entry.iloc[probs].to_csv('../data/intrusions.zip', index=False, compression=compression_opts) 

        


