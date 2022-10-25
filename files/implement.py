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
Author: Mark Harvey
'''


import os, sys
import argparse

from config import config as cfg

# indicate which GPU to use
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.gpu_list

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'


import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Vitis-Ai quantizer
from tensorflow_model_optimization.quantization.keras import vitis_quantize

from build_mobilenetv2 import build_mobilenetv2
from dataset_utils import input_fn_train, input_fn_test


# configuration
train_init_lr = cfg.train_init_lr
train_epochs = cfg.train_epochs
tfrec_dir = cfg.tfrec_dir
batchsize=cfg.batchsize
input_shape=cfg.input_shape
model_name=cfg.model_name
DIVIDER=cfg.DIVIDER


 

def train(model,output_ckpt,learnrate,train_dataset,test_dataset,epochs,batchsize):

  def step_decay(epoch):
    '''
    Learning rate scheduler used by callback
    Reduces learning rate depending on number of epochs
    '''
    lr = learnrate
    if epoch > int(epochs*0.9):
        lr /= 100
    elif epoch > int(epochs*0.05):
        lr /= 10
    return lr



  '''
  Call backs
  '''
  chkpt_call = ModelCheckpoint(filepath=output_ckpt,
                               monitor='val_accuracy',
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=True)

  lr_scheduler_call = LearningRateScheduler(schedule=step_decay,
                                            verbose=1)

  callbacks_list = [chkpt_call, lr_scheduler_call]
  

  '''
  Compile model
  '''
  model.compile(optimizer=Adam(learning_rate=learnrate),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  '''
  Training
  '''
  print('\n'+DIVIDER)
  print(' Training model with training set..')
  print(DIVIDER)


  # run training
  train_history=model.fit(train_dataset,
                          epochs=epochs,
                          steps_per_epoch=20000//batchsize,
                          validation_data=test_dataset,
                          validation_steps=None,
                          callbacks=callbacks_list,
                          verbose=0)

  return


def evaluate(model,test_dataset):
  '''
  Evaluate a keras model with the test dataset
  '''
  model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  scores = model.evaluate(test_dataset,
                          steps=None,
                          verbose=0)
  return scores



def implement(build_dir, mode, target):
  '''
  Implements training, quantization and compiling
  '''

  # output checkpoints and folders for each mode
  train_output_ckpt = build_dir + cfg.train_output_ckpt
  quant_output_ckpt = build_dir + cfg.quant_output_ckpt
  compile_output_dir = build_dir+cfg.compile_dir


  '''
  tf.data pipelines
  '''
  train_dataset = input_fn_train(tfrec_dir,batchsize)
  test_dataset = input_fn_test(tfrec_dir,batchsize)


  if (mode=='train'):

    # build mobilenet without weights
    model = build_mobilenetv2(input_shape=input_shape)

    # make folder for saving trained model checkpoint
    os.makedirs(os.path.dirname(train_output_ckpt), exist_ok=True)

    # run initial training
    train(model,train_output_ckpt,train_init_lr,train_dataset,test_dataset,train_epochs,batchsize)

    # eval trained checkpoint
    model = build_mobilenetv2(weights=train_output_ckpt, input_shape=input_shape)
    scores = evaluate(model,test_dataset)
    print('Trained model accuracy: {0:.4f}'.format(scores[1]*100),'%')


  elif (mode=='quantize'):

    # make folder for saving quantized model
    os.makedirs(os.path.dirname(quant_output_ckpt), exist_ok=True)

    # load the trained model
    float_model = build_mobilenetv2(weights=train_output_ckpt,input_shape=input_shape)

    # run quantization with fast fine-tune
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=test_dataset)

    # saved quantized model
    quantized_model.save(quant_output_ckpt)
    print('Saved quantized model to',quant_output_ckpt)

    '''
    Evaluate quantized model
    '''
    print('\n'+DIVIDER)
    print ('Evaluating quantized model..')
    print(DIVIDER+'\n')

    quantized_model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

    scores = quantized_model.evaluate(test_dataset,
                                      verbose=0)

    print('Quantized model accuracy: {0:.4f}'.format(scores[1]*100),'%')
    print('\n'+DIVIDER)


  elif (mode=='compile'):

    # set the arch value for the compiler script
    arch_dict = {
      'vck5000-4pe-miscdwc': 'DPUCVDX8H/VCK50004PE',
      'vck5000-6pedwc': 'DPUCVDX8H/VCK50006PEDWC',
      'vck5000-6pemisc': 'DPUCVDX8H/VCK50006PEMISC',
      'vck5000-8pe': 'DPUCVDX8H/VCK50008PE'
    }

    arch='/opt/vitis_ai/compiler/arch/'+arch_dict[target.lower()]+'/arch.json'
    print(' Compiling for',arch)
  
    # path to TF2 compiler script (valid for Vitis-AI 2.5)
    compiler_path = '/opt/vitis_ai/conda/envs/vitis-ai-tensorflow2/lib/python3.7/site-packages/vaic/vai_c_tensorflow2.py'

    # arguments for compiler script
    cmd_args = ' --model ' + quant_output_ckpt + ' --output_dir ' + compile_output_dir + ' --arch ' + arch + ' --net_name ' + model_name+'_'+target
    print('Compiler command:',cmd_args)

    # run compiler python script
    os.system(compiler_path + cmd_args)


  else:
    print('INVALID MODE - valid modes are train, quantize, compile')


  return



def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-bd', '--build_dir', type=str, default='build', help='Path of build folder. Default is build')
  ap.add_argument('-m',  '--mode',      type=str, default='train',        choices=['train','quantize','compile'], help='Mode: train,quantize,compile. Default is train')
  ap.add_argument('-t' , '--target',    type=str, default='vck5000-4pe',  choices=['vck5000-4pe-miscdwc','vck5000-6pedwc','vck5000-6pemisc','vck5000-8pe'], help='Target platform. Default is vck5000-4pe')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('Keras version      : ',tf.keras.__version__)
  print('TensorFlow version : ',tf.__version__)
  print(sys.version)
  print(DIVIDER)


  implement(args.build_dir, args.mode, args.target)


if __name__ == '__main__':
    run_main()
