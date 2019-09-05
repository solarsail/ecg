#!/bin/bash

if [ -z "$TF" ]
then
    TF=tensorflow
else
    TF=tensorflow-gpu
fi


pip2 install -r requirements.txt
pip2 install --upgrade $TF==1.8.0

