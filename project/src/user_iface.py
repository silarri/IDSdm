from intrusion_detector import *

data_handler = DATA_HANDLER("../data")
data_handler.load_data(verbose=True)


intrusion_detection_system = intrusion_detector(data_handler,["LOGREG","KNN"],0.90)

aux_X, aux_y = data_handler.get_train_data()
aux_X = aux_X.iloc[0:3]

intrusion_detection_system.find_intrusions(aux_X)


