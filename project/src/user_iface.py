from intrusion_detector import *

data_handler = DATA_HANDLER("../data")
data_handler.load_data(verbose=True)

intrusion_detection_system = intrusion_detector(data_handler,["LOGREG","KNN"],0.90)

#For simplicity
aux_x_test = data_handler.get_test_data()
aux_x_test = aux_x_test.iloc[0:20]

#Will use the test data provided in the data folder
intrusion_detection_system.find_intrusions(aux_x_test)


