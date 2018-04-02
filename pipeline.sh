#!/bin/bash -x

'''
batch move and label collected data
then create and train a model on this data
'''

# SET THESE VARIABLES. car should be an entry in ~/.ssh/config too
#name of the vehicle the model will be deployed to
name=$1
#which button you press on the DS4 to give the model control of the car
button=$2
#created from the name of the car refers to where the model will be deployed to 
car=${name%%-*}
#location where the data can be pulled from to the local train folder
data_source=$3

# Copy over training data from car
rsync --size-only -rvP $data_source* ${DERP_DATA}/train

# label data in the local train folder if it doesn't have a label file
for f in $DERP_DATA/train/*
do
    if ! [[ -e $f/label.csv ]]
    then
	python3 label.py --path $f
    fi
done

# Delete existing trained name
rm -rf ${DERP_SCRATCH}/${name}

# Train a new model
cd derp
python3 clone_create.py --config $name
python3 clone_train.py --config $name

# Deploy Model to the target vehicle
if [[ -n $button ]] ; then
    model=$(ls ${DERP_SCRATCH}/${name}/*pt | tail -n 1)
    rsync -rvP $model ${car}:${DERP_MODEL}/${button}/clone.pt
    if [[ $? --eq "0" ]] ; then
	   echo "SUCCESS"
    fi
fi
