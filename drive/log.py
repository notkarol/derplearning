#!/usr/bin/env python3

import os
from socket import gethostname
from time import strftime, gmtime

class Log:
    
    def __init__(self, screen, root_path="../data"):
        """
        Open the log files and write some logs
        """
        self.screen = screen
    
        # Create folder
        hostname = gethostname()
        self.date = strftime('%Y%m%dT%H%M%SZ', gmtime())
        self.name = '%s-%s' % (self.date, hostname)
        self.folder = os.path.join(root_path, self.name)
        os.mkdir(self.folder)

        # Record that we have a model
        self.config_path = os.path.join(self.folder, "config.txt")
        self.config_fp = open(self.config_path, 'w')

        # Create recording csv
        self.csv_path = os.path.join(self.folder, "video.csv")
        self.csv_fp = open(self.csv_path, 'w')
        
        # Write headers
        self.csv_fp.write("timestamp,speed,nn_speed,steer,nn_steer\n")


    def __del__(self):
        """
        Deconstructor to close file pointers
        """
        self.config_fp.flush()
        self.config_fp.close()
        self.csv_fp.flush()
        self.csv_fp.close()

        
    def config(self, key, value):
        """
        Update the config file with the key and value
        """
        self.config_fp.write("%s=%s\n" % (key, value))
        

    def write(self, args):
        """
        Write he provided values to a file
        """
        out = ','.join(['' if arg is None else "%.6f" % arg for arg in args])
        self.csv_fp.write(out + "\n")
