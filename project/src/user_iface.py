from intrusion_detector import *

def print_welcome():
    print ("Welcome to IDS-NET (Intrusion Detection System for Network comunications)\n")
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
algorithms = input("Insert a list separated by spaces of the algorithms you wish to use (LOGREG,KNN,DTREE): ")
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

#CALL THE IDS CONSTRUCTOR:
try:
    intrusion_detection_system = intrusion_detector(data_handler,algorithms,confidence_threshold)
except RuntimeError:
    print_exit(error=4)
    exit(1)

#TRAIN the system and PREDICT for the test data
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
