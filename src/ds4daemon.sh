#!/bin/bash -x
source ~/.derprc
cd $DERP_CODE
python3 ds4daemon.py >> /tmp/ds4daemon.log
