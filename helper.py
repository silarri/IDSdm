#Library to load network data
import os
import numpy as np
import pandas as pd


#######main###########

df_train = pd.read_csv("/home/arturo/Uni/4º/TFG/Tuesday-WorkingHours.pcap_ISCX.csv")
df_train.loc[df_train["class"] != "BENIGN", "class"] = "1" #1 MEANS INTRUSSION
df_train.loc[df_train["class"] == "BENIGN", "class"] = "0" #0 MEANS NORMAL
df_train.to_csv("/home/arturo/Uni/4º/TFG/SSH_FTP_ISCX.csv", index=False) 