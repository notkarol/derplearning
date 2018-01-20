#!/bin/bash -x
source ~/.derprc
cd $DERP_CODE
python3 daemon.py >> /tmp/ds4daemon.log
