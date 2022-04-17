INSTRUCTIONS TO USE IDS_1.0 IN DAEMON MODE:

1 - As the root user create the directory /etc/ids

2 - In such directory, always as the root user, copy the following files:
    TFG/app2/data/test/SSH_FTP_ISCX_test.csv
    TFG/app2/data/test/SSH_FTP_ISCX_train.csv
    TFG/app2/src/ids_1.0.py

3 - Edit the following global control variables in the file ids_1.0.py as needed:
    "INTERFACE" => Network interface to be used for analysis
    "CONFIDENCE_THRESHOLD" => Confidence threshold for the predictions

4 - As the root user copy the following file in the directory /etc/systemd/system:
    TFG/app2/src/ids-daemon.service

5 - Reload the systemd units and start the service with the following commands:
    sudo systemctl daemon-reload
    sudo systemctl enable ids-daemon
    sudo systemctl start ids-daemon
    sudo systemctl status ids-daemon

6 - The daemon will email the administrator when suspicius network traffic is identified.
    Such traffic will be logged in the file /etc/ids/ids_intrusions.csv


INSTRUCTIONS TO USE IDS_1.0 IN DEBUG MODE:

1 - As the root user create the directory /etc/ids

2 - In such directory, always as the root user, copy the following files:
    TFG/app2/data/test/SSH_FTP_ISCX_test.csv
    TFG/app2/data/test/SSH_FTP_ISCX_train.csv
    TFG/app2/src/ids_1.0.py

3 - Edit the following global control variables in the file ids_1.0.py as needed:
    "INTERFACE" => Network interface to be used for analysis
    "CONFIDENCE_THRESHOLD" => Confidence threshold for the predictions

4- Execute the following command and some performance metrics will be displayed:
    sudo python3 /etc/ids/ids-1.0.py DEBUG

DEPENDENCIES:
- python 3.0 or higher
- sklearn library 
- numpy library
- pandas library
- Sniffing command line application from the canadian institute of cibersecurity:
  https://gitlab.com/hieulw/cicflowmeter/-/tree/master

CONFIGURATION FOR CICFLOWMETER as of 17th April 2022:
1 - Download the code from the provided link
2 - In the file cicflowmeter/src/cicflowmeter/flow_session.py change the line 86 to the following line:
    flow.add_packet(packet, direction)
3 - Replace the file cicflowmeter/src/cicflowmeter/sniffer.py with the one provided in this directory
4 - Install the dependencies listed in the file requirements.txt with pip3
5 - Build the code ececuting the following command in the cicflowmeter directory:
    python setup.py install


