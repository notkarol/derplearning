#!/bin/bash -x

paths="opencv opencv_extra pytorch vision"

for path in $paths
do
    if [[ -e $path ]] && [[ -d $path ]]
    then
	rm -rf $path
    fi
done
