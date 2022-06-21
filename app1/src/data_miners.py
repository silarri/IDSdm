"""
Author: Arturo Calvera Tonin
Date: June 2022
Project: TFG - Data mining for intrusion detection in communication networks
File: data_miners.py
Comms: Library with data miners to perform machine learning on network data
"""
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

#Comms: Parent class to all data-miners / data mining algorithms
class DATA_MINER():
    #pca_rfe=0 => No feature selection,pca_rfe=1 => Performs PCA, pca_rfe=2 => Performs RFE
    #n_features => Number of features to use after feature selection (pca_rfe)
    def __init__(self,pca_rfe=0,n_features=10):
        self.model=None
        self.pca_rfe=pca_rfe
        self.n_features=n_features
        self.pretty_name = "Data-Miner"
        self.full_name = "Data-Miner"
    
    #Trains the algo and performs K fold cross validation algorithm with the data provided on X and y
    def K_fold_cross_val(self,X,y,splits,verbose=True):
        
        if verbose: print()
        aux_model = self.model 
        X = X.to_numpy()

        #Perform PCA or RFE:
        match self.pca_rfe:

            case 1: #PCA
                pca = PCA(n_components=self.n_features)
                pca.fit(X)
                X=pca.transform(X) #Apply dimensional reduction to input data

            case 2: #RFE
                aux_model = RFE(self.model, n_features_to_select=self.n_features,step=5,verbose=0)

        #Perform K-fold cross validation and obtain metrics
        metrics = {"Accuracy" : [], "Precision" : [], "Recall" : [], "F1score" : []}

        n_fold=0
        kf = StratifiedKFold(shuffle=True,n_splits=splits)

        if verbose: print(self.pretty_name+" :",end='',flush=True)

        for train_index, test_index in kf.split(X,y):
            if verbose: print("=",end='',flush=True)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #Train model and predict (every time .fit is called the model "resets")
            aux_model.fit(X_train,y_train)
            y_pred = aux_model.predict(X_test) 

            #Compute metrics
            #'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label)
            #p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted') 
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred,labels=[0, 1]).ravel()
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f = 2 * (p*r / (p+r))
            score = accuracy_score(y_test,y_pred)

            #plot_confusion_matrix(aux_model,X_test,y_test)
            #plt.show()
            #print(score,p,r,f)

            metrics["Accuracy"].append(score)
            metrics["Precision"].append(p)
            metrics["Recall"].append(r)
            metrics["F1score"].append(f)

        if verbose: print(">")

        return metrics

    #Trains the model with all the avaiable data
    def train_model(self,X,y): 
        #Perform PCA or RFE:
        match self.pca_rfe:
            case 0: #No feature selection is applied
                self.model.fit(X,y)
            case 1: #PCA
                pca = PCA(n_components=self.n_features)
                pca.fit(X)
                X=pca.transform(X) #Apply dimensional reduction to x_train
                self.pca_model=pca
                self.model.fit(X,y)
            case 2: #RFE
                rfe = RFE(self.model, n_features_to_select=self.n_features,step=15,verbose=0)
                self.model = rfe.fit(X, y)

        #print(x_train.shape,x_test.shape

    def transform_data_with_pca(self,X):
        return self.pca_model.transform(X)

    #predicts the probability that a network transmission is NOT an intrusion
    #Class == 1 => Intrusion, Class == 0 => Normal
    def predict_proba_intrusion(self,x):
        aux = self.model.predict_proba(x) #returns firs probability of normal and then intrusion in a row
        aux = np.asarray(aux)
        prob_normal = aux.transpose()[1]
        return prob_normal

    def predict_proba_normal(self,x):
        aux = self.model.predict_proba(x) #returns firs probability of normal and then intrusion in a row
        aux = np.asarray(aux)
        prob_intrusion = aux.transpose()[0]
        return prob_intrusion
    
    def predict(self,x):
        return self.model.predict(x)
    
    def pretty_name_helper(self,pca_rfe,n_features):
        if pca_rfe == 1:
            return "PCA-"+str(n_features)
        elif pca_rfe == 2:
            return "RFE-"+str(n_features)
        return ""


#Logistic regression
class LOGREG(DATA_MINER):

    def __init__(self,pca_rfe=0,n_features=10):
        super().__init__(pca_rfe,n_features)
        self.model=LogisticRegression(random_state=0,max_iter=3000)
        self.pretty_name = "Logistic_Regression"
        self.full_name = self.pretty_name + super().pretty_name_helper(self.pca_rfe,self.n_features)

#K nearest neighbour
class KNN(DATA_MINER):  

    def __init__(self,n_neighbors_=3,pca_rfe=0,n_features=10):
        super().__init__(pca_rfe,n_features)
        self.n_neighbors_=n_neighbors_
        if self.pca_rfe == 2: #RFE cannot be performed for KNN (There's no feature importance)
            self.pca_rfe = 1 
        self.model=KNeighborsClassifier(n_neighbors=self.n_neighbors_)
        self.pretty_name = "K_Nearest_Neighbour"
        self.full_name = self.pretty_name + super().pretty_name_helper(self.pca_rfe,self.n_features)
    
    #Tunes the number of neighbours to use (k), accros the range [start,stop] with step
    #Uses k fold cross validation with as many splits as indicated in splits
    #VERY TIME CONSUMING AND RESOURCE INTENSIVE
    #def tune_hyperparameter_k(self,X,y,start,stop,step,splits,verbose=False):
    #    ks = np.arange(start,stop,step).tolist()
    #    all_scores = []
    #    X = X.to_numpy()
    #    kf = KFold(shuffle=True,n_splits=splits)
    #    for train_index, test_index in kf.split(X):
    #        if verbose: print(":",end='',flush=True)
    #        X_train, X_test = X[train_index], X[test_index]
    #        y_train, y_test = y[train_index], y[test_index]
    #        scores=[]
    #        for k in ks:
    #            if verbose: print("=",end='',flush=True)
    #            model = KNeighborsClassifier(n_neighbors=k)           
    #            model.fit(X_train,y_train)
    #            scores.append(model.score(X_test,y_test))
    #        all_scores.append(scores)
    #        if verbose: print(">")
    #    all_scores = np.asarray(all_scores)
    #    all_scores = all_scores.mean(axis=0)
    #    return ks,all_scores

#Decision Tree
class DTREE(DATA_MINER):

    def __init__(self,pca_rfe=0,n_features=10):
        super().__init__(pca_rfe,n_features)
        self.model=DecisionTreeClassifier()
        self.pretty_name = "Decision_Tree"
        self.full_name = self.pretty_name + super().pretty_name_helper(self.pca_rfe,self.n_features)

#Gaussian Naive Bayes
class GNB(DATA_MINER):

    def __init__(self,pca_rfe=0,n_features=10):
        super().__init__(pca_rfe,n_features)
        if self.pca_rfe == 2: #RFE cannot be performed for GNB (There's no feature importance)
            self.pca_rfe = 1 
        self.model=GaussianNB()
        self.pretty_name = "Gaussian_Naive_Bayes"
        self.full_name = self.pretty_name + super().pretty_name_helper(self.pca_rfe,self.n_features)

#Multilayer Perceptron clasifier
class MLPC(DATA_MINER):

    def __init__(self,pca_rfe=0,n_features=10):
        super().__init__(pca_rfe,n_features)
        if self.pca_rfe == 2: #RFE cannot be performed for MLP (There's no feature importance)
            self.pca_rfe = 1 
        # 3 hidden layers
        self.model=MLPClassifier(random_state=0, max_iter=300, hidden_layer_sizes=(50,25,10,),verbose=False)
        self.pretty_name = "Multi-Layer_Perceptron_Classifier"
        self.full_name = self.pretty_name + super().pretty_name_helper(self.pca_rfe,self.n_features)
