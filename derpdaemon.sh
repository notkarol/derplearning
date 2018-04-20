#!/bin/bash -x
source ~/.derprc
cd $DERP_ROOT # TODO have derpdaemon be in $PATH
python3 derpdaemon.py >> /tmp/derpdaemon.log
