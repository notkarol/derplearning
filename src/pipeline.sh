#!/bin/bash -x

# SET THESE VARIABLES. car should be an entry in ~/.ssh/config too
name=$1
button=$2
car=${name%%-*}

# Copy over training data from car and label it
rsync -rvP ${car}:/mnt/sdcard/data/* ${DERP_DATA}/train
for f in $DERP_DATA/train/*
do
    if ! [[ -e $f/label.csv ]]
    then
	python3 label.py --path $f
    fi
done

# Delete existing trained name and train a new one
rm -rf ${DERP_SCRATCH}/${name}
python3 clone_create.py --car $name
python3 clone_train.py --car $name

# Deploy Model
if [[ -n $button ]] ; then
    model=$(ls ${DERP_SCRATCH}/${name}/*pt | tail -n 1)
    rsync -rvP $model ${car}:${DERP_MODEL}/${button}/clone.pt
    if [[ $? --eq "0" ]] ; then
	echo "SUCCESS"
    fi
fi
