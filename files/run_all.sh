#!/bin/sh

# Copyright 2022 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Mark Harvey


# Build script


# enable TensorFlow2 environment
conda activate vitis-ai-tensorflow2


echo "-----------------------------------------"
echo " STEP #0: SET UP ENVIRONMENT VARIABLES"
echo "-----------------------------------------"

# make build folder
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}



echo "-----------------------------------------"
echo " STEP #1: CONVERT DATASET TO TFRECORDS"
echo "-----------------------------------------"
python -u images_to_tfrec.py 2>&1 | tee ${LOG}/tfrec.log


echo "-----------------------------------------"
echo " STEP #2: TRAINING"
echo "-----------------------------------------"
python -u implement.py --mode train --build_dir ${BUILD} 2>&1 | tee ${LOG}/train.log



echo "-----------------------------------------"
echo " STEP #3: QUANTIZATION"
echo "-----------------------------------------"
python -u implement.py --mode quantize --build_dir ${BUILD} 2>&1 | tee ${LOG}/quantize.log


echo "-----------------------------------------"
echo " STEP #4: COMPILE FOR TARGET"
echo "-----------------------------------------"
# modify the list of targets as required
for targetname in vck5000-4pe-miscdwc vck5000-6pedwc; do
  python -u implement.py --mode compile --build_dir ${BUILD} --target ${targetname} 2>&1 | tee ${LOG}/compile_${targetname}.log
done


echo "-----------------------------------------"
echo " STEP #5: MAKE TARGET FOLDER"
echo "-----------------------------------------"
python -u target.py --build_dir  ${BUILD} 2>&1 | tee ${LOG}/target.log


echo "-----------------------------------------"
echo " FLOW COMPLETED.."
echo "-----------------------------------------"

