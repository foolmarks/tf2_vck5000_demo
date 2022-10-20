'''
 Copyright 2022 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''

'''
Configuration
'''

'''
Author: Mark Harvey
'''



DIVIDER = '-----------------------------------------'


# GPU
gpu_list='0'

# Dataset preparation and TFRecord creation
data_dir = 'data'
tfrec_dir = data_dir + '/tfrecords'
img_shard = 500


# MobileNet build parameters
input_shape=(224,224,3)
classes=2
alpha=1.0


# Training
batchsize=150
train_init_lr=0.001
train_epochs=85
train_target_acc=1.0
train_output_ckpt='/float_model/f_model.h5'

# Quantization
quant_output_ckpt='/quant_model/q_model.h5'

# Compile
compile_dir='/compiled_model_'
model_name='mobilenetv2'

# Target
target_dir='/target_'

# Application code
app_dir='application'

