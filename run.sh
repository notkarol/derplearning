#!/bin/bash -x

for i in $(seq $2) 
do
  python3 bin/drive.py $1
done
