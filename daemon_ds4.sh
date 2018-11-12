#!/bin/bash
source ~/.derprc
cd $DERP_ROOT
python3 daemon_ds4.py >> /tmp/dameon_ds4.out 2>&1
