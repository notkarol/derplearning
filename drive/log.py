#!/usr/bin/env python3

import cv2
import os
from socket import gethostname
from time import strftime, gmtime

class Log:
    
    def __init__(self, root_path="."):
        """
        Open the log files and write some logs
        """

        # Create folder
        self.folder = os.path.join(root_path, strftime('%Y%m%dT%H%M%SZ', gmtime()))
        os.mkdir(self.folder)

        # Record that we have a model
        self.config_path = os.path.join(self.folder, "config.txt")
        self.config_fp = open(self.config_path, 'w')

        # Create recording csv
        self.csv_path = os.path.join(self.folder, "video.csv")
        self.csv_fp = open(self.csv_path, 'w')
        
        # Write headers
        self.csv_fp.write("timestamp,speed,steer\n")

        # Save some attributes about the machine
        self.config('hostname', gethostname())
        self.config('cv2_version', cv2.__version__)
        

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
        

    def write(self, timestamp, speed, steer):
        """
        Write he provided values to a file
        """
        timestamp_str = "%.6f" % timestamp
        speed_str = "%.6f" % speed
        steer_str = "%.6f" % steer

        # Create video
        self.csv_fp.write(",".join([timestamp_str, speed_str, steer_str]) + "\n")
