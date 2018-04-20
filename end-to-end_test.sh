#!/bin/bash -x

# Tests the data pipeline script 
# on a small slice of data to verify minimum project functionality

#_________________setting up the test data___________________
#variables
data_size=32
#choses the last folder added to data to use as the test case
DATA_SOURCE=$(ls -td $DERP_ROOT/data/2* | head -1)
DATA_NAME=short_slice
TEST_DATA_FOLDER=test_train
TEST_DIR=$DERP_ROOT/data/$TEST_DATA_FOLDER
TEST_CONFIG=test_config
BUTTON=triangle

#delete previous test data if any exists
rm -rf $TEST_DIR/*
#delete previous test config if any exists
rm $DERP_ROOT/config/${TEST_CONFIG}.yaml

#create the test data directory and it's appropriate contents:
mkdir $TEST_DIR/$DATA_NAME
mkdir $TEST_DIR/$DATA_NAME/camera_front

# copy over the recorded data vehicle config file
cp $DATA_SOURCE/config.yaml $TEST_DIR/$DATA_NAME
# copy over the video file (uncropped)
cp $DATA_SOURCE/*.mp4 $TEST_DIR/$DATA_NAME
# Cut a small slice of data to prepare it for training moving it to a test folder
head -$data_size ${DATA_SOURCE}/state.csv > $TEST_DIR/$DATA_NAME/state.csv
# copy the selected number of image frames over to the test data folder
for image in $(ls $DATA_SOURCE/camera_front | head -n $data_size)
do
    cp $DATA_SOURCE/camera_front/$image $TEST_DIR/$DATA_NAME/camera_front/
done

# copy the model config file renaming it as the test config file
cp $DERP_ROOT/config/paras.yaml $DERP_ROOT/config/${TEST_CONFIG}.yaml
# edit the model config file to point to the test dataset
sed -i "s/'train'/'test_train'/g" $DERP_ROOT/config/${TEST_CONFIG}.yaml

#_____________test the pipeline.sh___________________

. $DERP_ROOT/pipeline.sh $TEST_CONFIG $TEST_DIR


#_________________test drive.py______________________

