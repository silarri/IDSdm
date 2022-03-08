import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

# TODO update pretty name...
#Parent class for all data mining algorithms
class DATA_MINER():
    def __init__(self,pca_rfe=0,n_features=10):
        self.model=None
        self.pca_rfe=pca_rfe
        self.n_features=n_features
        self.pretty_name = "Data-Miner"
        self.full_name = "Data-Miner"
    
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
                aux_model = RFE(self.model, n_features_to_select=self.n_features,step=15,verbose=0)

        #Perform K-fold cross validation and obtain metrics
        metrics = {"Accuracy" : [], "Precision" : [], "Recall" : [], "Fscore" : []}

        n_fold=0
        kf = KFold(shuffle=True,n_splits=splits)

        if verbose: print(self.pretty_name+" :",end='',flush=True)

        for train_index, test_index in kf.split(X):
            if verbose: print("=",end='',flush=True)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #Train model and predict (every time .fit is called the model "resets")
            aux_model.fit(X_train,y_train)
            y_pred = aux_model.predict(X_test) 

            #Compute metrics
            #'weighted': Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label)
            p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted') 
            score = aux_model.score(X_test,y_test)

            metrics["Accuracy"].append(score)
            metrics["Precision"].append(p)
            metrics["Recall"].append(r)
            metrics["Fscore"].append(f)

        if verbose: print(">")

        return metrics

    #trains the model with all the avaiable data
    def train_model(self,X,y): 
        #Perform PCA or RFE:
        match self.pca_rfe:
            case 0: #nothing special is done
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
    def predict_proba_normal(self,x):
        aux = self.model.predict_proba(x) #returns firs probability of intrusion and then normal in a row
        aux = np.asarray(aux)
        prob_normal = aux.transpose()[1]
        return prob_normal

    def predict_proba_intrusion(self,x):
        aux = self.model.predict_proba(x) #returns firs probability of intrusion and then normal in a row
        aux = np.asarray(aux)
        prob_intrusion = aux.transpose()[0]
        return prob_intrusion


#Logistic regression
class LOGREG(DATA_MINER):

    def __init__(self,pca_rfe=0,n_features=10):
        super().__init__(pca_rfe,n_features)
        self.model=LogisticRegression(random_state=0,max_iter=3000)
        self.pretty_name = "Logistic Regression"
        self.full_name = self.pretty_name +"_"+ str(pca_rfe) + "_"+str(n_features)

#K nearest neighbour
class KNN(DATA_MINER):  

    #TODO: optimally choose K
    def __init__(self,n_neighbors_=5,pca_rfe=0,n_features=10):
        super().__init__(pca_rfe,n_features)
        self.n_neighbors_=n_neighbors_
        if self.pca_rfe == 2: #RFE cannot be performed for KNN (There's no feature importance)
            self.pca_rfe = 1 
        self.model=KNeighborsClassifier(n_neighbors=self.n_neighbors_)
        self.pretty_name = "K Nearest Neighbour"
        self.full_name = self.pretty_name +"_"+ str(pca_rfe) + "_"+str(n_features)

#Decision Tree
class DTREE(DATA_MINER):

    def __init__(self,pca_rfe=0,n_features=10):
        super().__init__(pca_rfe,n_features)
        self.model=DecisionTreeClassifier()
        self.pretty_name = "Decision Tree"
        self.full_name = self.pretty_name +"_"+ str(pca_rfe) + "_"+str(n_features)

#Gaussian Naive Bayes
class GNB(DATA_MINER):

    def __init__(self,pca_rfe=0,n_features=10):
        super().__init__(pca_rfe,n_features)
        if self.pca_rfe == 2: #RFE cannot be performed for GNB (There's no feature importance)
            self.pca_rfe = 1 
        self.model=GaussianNB()
        self.pretty_name = "Gaussian Naive Bayes"
        self.full_name = self.pretty_name +"_"+ str(pca_rfe) + "_"+str(n_features)

#Multilayer Perceptron clasifier
class MLPC(DATA_MINER):

    def __init__(self,pca_rfe=0,n_features=10):
        super().__init__(pca_rfe,n_features)
        if self.pca_rfe == 2: #RFE cannot be performed for MLP (There's no feature importance)
            self.pca_rfe = 1 
        # 3 hidden layers
        self.model=MLPClassifier(random_state=0, max_iter=300, hidden_layer_sizes=(50,25,10,),verbose=False)
        self.pretty_name = "Multi-Layer Perceptron Classifier"
        self.full_name = self.pretty_name +"_"+ str(pca_rfe) + "_"+str(n_features)
        
'''

#Class for holding train and test data
class DATA_HOLDER():

    def __init__(self,X,y):
        self.X,self.y = X,y
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y)

    def return_data(self):
        return self.X, self.y, self.x_train, self.x_test, self.y_train, self.y_test

#Parent class for all classifiers
class CLASSIFIER():
    def __init__(self,X,y,x_train,x_test,y_train,y_test,with_PCA=False,n_components=0):
        self.X,self.y = shuffle(X,y,random_state=0)
        self.x_train, self.x_test, self.y_train, self.y_test = x_train,x_test,y_train,y_test
        self.model=None
        self.with_PCA=with_PCA
        self.components=n_components

    def test_model(self):
        return self.model.score(self.x_test,self.y_test)
        #Return the mean accuracy on the given test data and labels.
    
    def plot_conf_mat(self):
        ConfusionMatrixDisplay.from_estimator(self.model,self.x_test,self.y_test)
        plt.show()
        plt.close('all')
    
    def return_metrics(self):
        cfm= confusion_matrix(self.y_test, self.model.predict(self.x_test))

        FP = cfm.sum(axis=0) - np.diag(cfm)  
        FN = cfm.sum(axis=1) - np.diag(cfm)
        TP = np.diag(cfm)
        TN = cfm.sum() - (FP+FN+TP)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        #PPV = TP/(TP+FP)
        # Negative predictive value
        #NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        #FDR = FP/(TP+FP)
        return TPR, TNR, FPR, FNR

    #Allows the introduction of records from impostors (outsider_samples)
    def plot_metrics_by_threshold(self,outsider_samples=[],thresholds = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
        
        def adjust_all_predictions(predicted_probs,threshold):
            return [1 if y >= threshold else 0 for y in predicted_probs] #1 => its the user 0=> not the user
        def adjust_ytest_to_user(ytest,user):
            return [1 if val == user else 0 for val in ytest] #1 => its the user 0=> not the user

        TPR_ = []
        TNR_ = [] 
        FPR_ = []
        FNR_ = []
        if  len(outsider_samples) != 0:
            X_test = np.concatenate((self.x_test, outsider_samples),axis=0)
            Y_test = np.concatenate((self.y_test,np.asarray([int(len(outsider_samples)+1) for i in range(len(outsider_samples))])),axis=0)
        else:
            X_test=self.x_test
            Y_test=self.y_test

        y_pred_proba =self.model.predict_proba(X_test)

        for threshold in thresholds:
            meanTPR = 0
            meanTNR = 0
            meanFPR = 0
            meanFNR = 0

            for user in self.model.classes_:
                y_test_aux=adjust_ytest_to_user(Y_test,user)
                y_pred_adj = adjust_all_predictions(y_pred_proba[:,user],threshold)

                cfm= confusion_matrix(y_test_aux, y_pred_adj) #new binary conf matt one vs all
                TN,FP,FN,TP = cfm.ravel()
                meanTPR += TP/(TP+FN)
                meanTNR += TN/(TN+FP)
                meanFPR += FP/(FP+TN)
                meanFNR += FN/(TP+FN)

            TPR_.append(meanTPR/len(self.model.classes_))
            TNR_.append(meanTNR/len(self.model.classes_))
            FPR_.append(meanFPR/len(self.model.classes_))
            FNR_.append(meanFNR/len(self.model.classes_))

        Y = [FPR_, FNR_, TPR_, TNR_]
        titles = ['False Acceptance Rate', 'False Rejection Rate', 'True Acceptance Rate', 'True Rejection Rate']
        plt.figure(figsize=(12,6))
        for y in Y:
            plt.plot(thresholds,y)
        plt.legend(titles)
        plt.xlabel('Treshold')
        plt.grid('on')
        plt.ylim([-0.2,1.2])
        plt.title('Evaluation rates')
        plt.show()
        

#Linear discriminant analisis model
class LDA(CLASSIFIER):

    def __init__(self,X,y,x_train,x_test,y_train,y_test,with_PCA=False,n_components=0):
        super().__init__(X,y,x_train,x_test,y_train,y_test,with_PCA,n_components)

    def train_model(self): 
        #linear discriminant analysis with optional dimensional reduction
        if self.with_PCA:
            #Perform PCA and extract the first int(components) principal components
            self.pca = PCA(n_components=self.components)
            self.pca.fit(self.X)
            #Apply dimensional reduction to x_train and x_test
            self.x_train=self.pca.transform(self.x_train)
            self.x_test=self.pca.transform(self.x_test)

        self.model=LinearDiscriminantAnalysis()
        self.model.fit(self.x_train,self.y_train)

#Logistic regression
class LOGREG(CLASSIFIER):

    def __init__(self,X,y,x_train,x_test,y_train,y_test,with_PCA=False,n_components=0):
        super().__init__(X,y,x_train,x_test,y_train,y_test,with_PCA,n_components)

    def train_model(self): 
        if self.with_PCA:
            #Perform PCA and extract the first int(components) principal components
            self.pca = PCA(n_components=self.components)
            self.pca.fit(self.X)
            #Apply dimensional reduction to x_train and x_test
            self.x_train=self.pca.transform(self.x_train)
            self.x_test=self.pca.transform(self.x_test)

        self.model=LogisticRegression(random_state=0,max_iter=3000)
        self.model.fit(self.x_train,self.y_train)

#K nearest neighbour
class KNN(CLASSIFIER):

    def __init__(self,X,y,x_train,x_test,y_train,y_test,with_PCA=False,n_components=0,n_neighbors_=5):
        super().__init__(X,y,x_train,x_test,y_train,y_test,with_PCA,n_components)
        self.n_neighbors_=n_neighbors_

    def train_model(self): 
        if self.with_PCA:
            #Perform PCA and extract the first int(components) principal components
            self.pca = PCA(n_components=self.components)
            self.pca.fit(self.X)
            #Apply dimensional reduction to x_train and x_test
            self.x_train=self.pca.transform(self.x_train)
            self.x_test=self.pca.transform(self.x_test)

        self.model=KNeighborsClassifier(n_neighbors=self.n_neighbors_)
        self.model.fit(self.x_train,self.y_train)

'''