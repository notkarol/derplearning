#!/bin/bash -x

# Tests the data pipeline script 
# on a small slice of data to verify minimum project functionality


#variables
data_size=32
#choses the last folder added to data to use as the test case
DATA_SOURCE=$(ls -td $DERP_DATA/2* | head -1)
DATA_NAME=dummy_data
TEST_DIR=test_data

#delete previous test data if any exists
rm -rf ./$TEST_DIR/$DATA_NAME*

#create the test data directory and it's appropriate contents:
mkdir ./$TEST_DIR/$DATA_NAME
mkdir ./$TEST_DIR/${DATA_NAME}/camera_front

# Cut a small slice of data to prepare it for training moving it to a test folder
# note: this does not crop or move the recorded video

head -$data_size ${DATA_SOURCE}/state.csv > ./$TEST_DIR/$DATA_NAME/state.csv


# test the pipeline script



# test drive.py

