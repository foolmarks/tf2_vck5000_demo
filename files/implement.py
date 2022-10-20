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


try:
  from tf_nndct.optimization import IterativePruningRunner
except:
  print('tf_nndct.optimization not found')

try:
  from tensorflow_model_optimization.quantization.keras import vitis_quantize
except:
  print('tensorflow_model_optimization.quantization.keras not found')

from build_mobilenetv2 import build_mobilenetv2
from dataset_utils import input_fn_train, input_fn_test


# configuration
train_target_acc = cfg.train_target_acc
train_init_lr = cfg.train_init_lr
train_epochs = cfg.train_epochs
tfrec_dir = cfg.tfrec_dir
batchsize=cfg.batchsize
input_shape=cfg.input_shape
model_name=cfg.model_name

init_prune_ratio=cfg.init_prune_ratio
incr_prune_ratio=cfg.incr_prune_ratio
prune_steps=cfg.prune_steps
finetune_init_lr=cfg.finetune_init_lr




DIVIDER=cfg.DIVIDER


class EarlyStoponAcc(tf.keras.callbacks.Callback):
  '''
  Early stop on reaching target accuracy 
  '''
  def __init__(self, target_acc):
    super(EarlyStoponAcc, self).__init__()
    self.target_acc=target_acc

  def on_epoch_end(self, epoch, logs=None):
    accuracy=logs['val_accuracy']
    if accuracy >= self.target_acc:
      self.model.stop_training=True
      print('Reached target accuracy of',self.target_acc,'..exiting.')
  

def train(model,output_ckpt,learnrate,train_dataset,test_dataset,epochs,batchsize,target_acc):

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

  early_stop_call = EarlyStoponAcc(target_acc)

  lr_scheduler_call = LearningRateScheduler(schedule=step_decay,
                                            verbose=1)

  callbacks_list = [chkpt_call, early_stop_call, lr_scheduler_call]

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


def ana_eval(model, test_dataset):
  return evaluate(model, test_dataset)[0]


def prune(model, ratio, test_dataset):
  '''
  Prune the model
  '''
  input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
  runner = IterativePruningRunner(model, input_spec)
  import functools
  eval_fn = functools.partial(ana_eval, test_dataset=test_dataset)
  runner.ana(eval_fn)
  return runner.prune(ratio)


def implement(build_dir, mode, target):
  '''
  Implements training, pruning and transform modes
  '''

  # output checkpoints and folders for each mode
  train_output_ckpt = build_dir + cfg.train_output_ckpt
  prune_output_ckpt = build_dir + cfg.prune_output_ckpt
  transform_output_ckpt = build_dir + cfg.transform_output_ckpt
  quant_output_ckpt = build_dir + cfg.quant_output_ckpt
  compile_output_dir = build_dir+cfg.compile_dir+target


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
    train(model,train_output_ckpt,train_init_lr,train_dataset,test_dataset,train_epochs,batchsize,train_target_acc)

    # eval trained checkpoint
    model = build_mobilenetv2(weights=train_output_ckpt, input_shape=input_shape)
    scores = evaluate(model,test_dataset)
    print('Trained model accuracy: {0:.4f}'.format(scores[1]*100),'%')

    # save final accuracy to a text file for use in pruning
    f = open(build_dir+'/trained_accuracy.txt', 'w')
    f.write(str(scores[1]))
    f.close()

  elif (mode=='prune'):

    # build mobilenet with weights from initial training
    model = build_mobilenetv2(weights=train_output_ckpt,input_shape=input_shape)

    prune_ratio=init_prune_ratio

    # fetch the required final accuracy
    f = open(build_dir+'/trained_accuracy.txt', 'r')
    final_ft_acc = float(f.readline())
    f.close()

    for i in range(1,prune_steps+1):

      print(DIVIDER)
      print('Pruning iteration',i,'of',prune_steps,' Pruning ratio:',prune_ratio)

      if (i==prune_steps):
        finetune_target_acc=final_ft_acc
      else:
        finetune_target_acc=final_ft_acc*0.97

      print('Target accuracy for this iteration:',finetune_target_acc)

      # prune model
      pruned_model = prune(model,prune_ratio,test_dataset)

      # fine-tune pruned model
      train(pruned_model,prune_output_ckpt,finetune_init_lr,train_dataset,test_dataset,train_epochs,batchsize,finetune_target_acc)

      # increment the pruning ratio for the next iteration
      prune_ratio+=incr_prune_ratio


    # eval best fine-tuned checkpoint
    model = build_mobilenetv2(weights=prune_output_ckpt, input_shape=input_shape)
    scores = evaluate(model,test_dataset)
    print('Pruned model accuracy: {0:.4f}'.format(scores[1]*100),'%')

  elif (mode=='transform'):

    # build mobilenet with weights from last pruning iteration
    model = build_mobilenetv2(weights=prune_output_ckpt,input_shape=input_shape)

    # make and save slim model
    input_spec = tf.TensorSpec((1, *input_shape), tf.float32)
    runner = IterativePruningRunner(model, input_spec) 
    slim_model = runner.get_slim_model()
    os.makedirs(os.path.dirname(transform_output_ckpt), exist_ok=True)
    slim_model.save(transform_output_ckpt)

    # eval slim model
    scores = evaluate(slim_model,test_dataset)
    print('Slim model accuracy: {0:.4f}'.format(scores[1]*100),'%')

  elif (mode=='quantize'):

    # make folder for saving quantized model
    os.makedirs(os.path.dirname(quant_output_ckpt), exist_ok=True)

    # load the transformed model if it exists
    # otherwise, load the trained model
    if (os.path.exists(transform_output_ckpt)):
      print('Loading transformed model..')
      float_model = load_model(transform_output_ckpt,compile=False)
    else:
      input_ckpt = train_output_ckpt
      print('Did not find transformed model, loading trained model..')
      # build mobilenet with weights
      float_model = build_mobilenetv2(weights=input_ckpt,input_shape=input_shape)

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
      'zcu102': 'DPUCZDX8G/ZCU102',
      'zcu104': 'DPUCZDX8G/ZCU104',
      'kv260': 'DPUCZDX8G/KV260',
      'u200': 'DPUCADF8H/U200',
      'u250': 'DPUCADF8H/U250',
      'u50': 'DPUCAHX8H/U50',
      'u50lv': 'DPUCAHX8H/U50LV',
      'u50lv-dwc': 'DPUCAHX8H/U50LV-DWC',
      'u55c': 'DPUCAHX8H/U55C-DWC',    
      'u280': 'DPUCAHX8L/U280',
      'vck190': 'DPUCVDX8G/VCK190',
      'vck5000-6pedwc': 'DPUCVDX8H/VCK50006PEDWC',
      'vck5000-8pe': 'DPUCVDX8H/VCK50008PE'
    }

    arch='/opt/vitis_ai/compiler/arch/'+arch_dict[target.lower()]+'/arch.json'
  
    # path to TF2 compiler script (valid for Vitis-AI 2.0)
    compiler_path = '/opt/vitis_ai/conda/envs/vitis-ai-tensorflow2/lib/python3.7/site-packages/vaic/vai_c_tensorflow2.py'

    # arguments for compiler script
    cmd_args = ' --model ' + quant_output_ckpt + ' --output_dir ' + compile_output_dir + ' --arch ' + arch + ' --net_name ' + model_name

    # run compiler python script
    os.system(compiler_path + cmd_args)


  else:
    print('INVALID MODE - valid modes are train, prune, transform, quantize, compile')


  return



def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-bd', '--build_dir', type=str, default='build', help='Path of build folder. Default is build')
  ap.add_argument('-m',  '--mode',      type=str, default='train', choices=['train','prune','transform','quantize','compile'], help='Mode: train,prune,transform,quantize,compile. Default is train')
  ap.add_argument('-t' , '--target',    type=str, default='zcu102',help='Target platform. Default is zcu102')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print('Keras version      : ',tf.keras.__version__)
  print('TensorFlow version : ',tf.__version__)
  print(sys.version)
  print(DIVIDER)


  implement(args.build_dir, args.mode, args.target)


if __name__ == '__main__':
    run_main()