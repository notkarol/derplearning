#!/bin/bash -x


#batch move and label collected data
#then create and train a model on this data


# SET THESE VARIABLES. car should be an entry in ~/.ssh/config too
#name of the vehicle the model will be deployed to
name=$1
#created from the name of the car refers to where the model will be deployed to 
car=${name%%-*}
#location where the data can be pulled from to the local train folder
data_source=$2
#which button you press on the DS4 to give the model control of the car
button=$3

#location where data is put for labling and training:
train_folder=$(python3 -c "import derp.util as util; print(util.pass_config(\"$DERP_ROOT/config/${name}.yaml\", 'components', 6, 'create', 'data_folders')[0] )")
#Note this will break when we split the model configs from the vehicle configs
train_loc=$DERP_ROOT/data/$train_folder


# Copy over training data from car
if $data_source != $train_loc
    then
    rsync --size-only -rvP $data_source/* $train_loc 
fi

# label data in the local train folder if it doesn't have a label file
for f in $train_loc/* 
do
    if ! [[ -e $f/label.csv ]]
    then
	python3 label.py --path $f --scale .8
    fi
done

# Delete existing trained name
rm -rf ${DERP_ROOT}/scratch/${name}

# Train a new model
python3 clone_build.py --config $name
python3 clone_train.py --config $name

# Deploy Model to the target vehicle
if [[ -n "$button" ]] ; then
    model=$(ls ${DERP_ROOT}/scratch/${name}/*pt | tail -n 1)
    rsync -rvP $model ${car}:${DERP_ROOT}/model/${button}/clone.pt
    if [[ $? -eq "0" ]] ; then
	   echo "SUCCESS"
    fi
fi
