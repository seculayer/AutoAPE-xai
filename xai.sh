#!/bin/bash
######################################################################################
# eyeCloudAI 3.1 MLPS Run Script
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2021 Service Model Team, R&D Center.
######################################################################################

APP_PATH=/eyeCloudAI/app/ape

XAI_LIB_PATH=$APP_PATH/xai/lib
XAI_PATH=$APP_PATH/xai
####
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=$XAI_LIB_PATH:$XAI_PATH

KEY=${1}
WORKER_IDX=${2}

/usr/local/bin/python3.7 -m xai.AutoAPEXAI ${KEY} ${WORKER_IDX}
