#!/bin/bash -x

# SET THESE VARIABLES. car should be an entry in ~/.ssh/config too
car=paras
button=triangle

# Copy over training data from car and label it
rsync -rvP ${car}:/mnt/sdcard/data/* ${DERP_DATA}/train
for f in $DERP_DATA/train/*
do
    if ! [[ -e $f/label.csv ]]
    then
	python3 label.py --path $f
    fi
done

# Delete existing trained model and train a new one
rm -rf ${DERP_SCRATCH}/${car}-*
python3 clone_create.py --car ${DERP_CONFIG}/${car}.yaml
python3 clone_train.py --car ${DERP_CONFIG}/${car}.yaml

# Copy over best trained model to car
path=$(ls ${DERP_SCRATCH} | grep ${car}-)
model=$(ls $path/*pt | tail -n 1)
rsync -rvP $model ${car}:/mnt/sdcard/${button}

echo "DONE"
