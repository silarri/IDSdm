import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE



#Parent class for all data mining algorithms
class DATA_MINER():
    def __init__(self,pca_rfe=0,n_features=10):
        self.model=None
        self.pca_rfe=pca_rfe
        self.n_features=n_features
    
    #used to estimate the skill of the algorithms (accuracy)
    #Use only in testbench
    def cross_val(self,X,y):
        #Perform PCA or RFE:
        match self.pca_rfe:
            case 0:
                scores = cross_val_score(self.model, X,y, cv=10)

            case 1: #PCA
                pca = PCA(n_components=self.n_features)
                pca.fit(X)
                X=pca.transform(X) #Apply dimensional reduction to x_train
                scores = cross_val_score(self.model, X,y, cv=10)

            case 2: #RFE
                model = RFE(self.model, n_features_to_select=self.n_features,step=15,verbose=0)
                scores = cross_val_score(model, X,y, cv=10)

        return scores.mean(), scores.std()

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

#K nearest neighbour
class KNN(DATA_MINER):  

    #TODO: optimally choose K
    def __init__(self,n_neighbors_=5,pca_rfe=0,n_features=10):
        super().__init__(pca_rfe,n_features)
        self.n_neighbors_=n_neighbors_
        if self.pca_rfe == 2: #RFE cannot be performed for KNN
            self.pca_rfe = 1 
        self.model=KNeighborsClassifier(n_neighbors=self.n_neighbors_)

#Support Vector machine classifier
class SVC_(DATA_MINER):
    def __init__(self,pca_rfe=0,n_features=10):
        super().__init__(pca_rfe,n_features)
        self.model=make_pipeline(StandardScaler(), SVC(probability=True,gamma='auto'))

class DTREE(DATA_MINER):
    def __init__(self,pca_rfe=0,n_features=10):
        super().__init__(pca_rfe,n_features)
        self.model=DecisionTreeClassifier()


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