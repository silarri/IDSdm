from intrusion_detector import *

def print_welcome():
    print ("Welcome to IDS-NET (Intrusion Detection System for Comunication Network)\n")
    print ("Please indicate the relative path to the folder containing the training and testing data.")
    print ("It MUST follow the following structure:\n")
    print ("folder/")
    print ("|-train/")
    print ("|   |-train_data.csv")
    print ("|-test/")
    print ("|   |-test_data.csv")
    path = input(":")
    return path

def print_exit(error=0):
    if error == 0:
        print("\nBYE!")
    elif error == 1:
        print("\nABORTING EXECUTION! Encountered a problem with the data directory!")
    elif error == 2:
        print("\nABORTING EXECUTION! No algorithm was introduced!")
    elif error == 3:
        print("\nABORTING EXECUTION! Threshold must be a decimal!")
    elif error == 4:
        print("\nABORTING EXECUTION! Number of features must be an integer")
    elif error == 5:
        print("\nABORTING EXECUTION! None of the algorithms introduced is avaiable")
#############################################  MAIN   ########################################

#print welcome message
path = print_welcome()

#Load the provided data
data_handler = DATA_HANDLER(path)
if not data_handler.load_data(verbose=True):
    print_exit(error=1)
    exit(1)

#Ask for the algorithms the intrusion detector is going to use:
algorithms = input("Insert a list separated by spaces of the algorithms you wish to use (LOGREG,KNN,DTREE,GNB): ")
algorithms = algorithms.split()
if not algorithms:
    print_exit(error=2)
    exit(1)

#Ask for a conficence threshold:
confidence_threshold = input("Insert a confidence threshold (eg 0.9): ")
try:
    confidence_threshold = float(confidence_threshold)
except ValueError:
    print_exit(error=3)
    exit(1)

#Ask the user if they want to use PCA or RFE
pca_rfe=input("Do you want to perfom PCA or RFE to the DATA?(PCA,RFE,NO)(RFE not avaiable with KNN & GNB):")
if pca_rfe == "PCA":
    pca_rfe=1
elif pca_rfe=="RFE":
    pca_rfe=2
else: #NONE
    pca_rfe=0

n_features=0
if pca_rfe == 1 or pca_rfe == 2:
    n_features = input("Insert the number of features to be used (int):")
    try:
        n_features = int(n_features)
    except ValueError:
        print_exit(error=4)
        exit(1)
    

#CALL THE IDS CONSTRUCTOR:
print("Please WAIT while training is performed")
try:
    intrusion_detection_system = intrusion_detector(data_handler,algorithms,confidence_threshold,pca_rfe,n_features)
except RuntimeError:
    print_exit(error=5)
    exit(1)

#TRAIN the system and PREDICT for the test data

print("Please WAIT for the results")
intrusion_detection_system.find_intrusions(path)
print("The file "+ os.path.join(path,"intrusions.csv")+" has been created.")
print("It contains the network conexions considered to be intrusions")
print_exit()

#For simplicity
#aux_x_test = data_handler.get_test_data()
#aux_x_test = aux_x_test.iloc[0:20]
#intrusion_detection_system.find_intrusions(aux_x_test)


#

#
##Will use the test data provided in the data folder
#intrusion_detection_system.find_intrusions(aux_x_test)
#
#
