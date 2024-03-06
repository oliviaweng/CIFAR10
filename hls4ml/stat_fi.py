import os
if os.system('nvidia-smi') == 0:
    import setGPU
import tensorflow as tf
import glob
import sys
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
import resnet_v1_eembc
import yaml
import csv
# from keras_flops import get_flops # (different flop calculation)
# import kerop
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop
random_crop_model = tf.keras.models.Sequential()
random_crop_model.add(RandomCrop(32, 32, input_shape=(32, 32, 3,)))

from tensorflow.keras.losses import CategoricalCrossentropy
from fkeras.metrics.hessian import HessianMetrics
from fkeras.metrics.stat_fi import StatFI
import time

import numpy as np

# Pickling Utils related imports ##############
import pickle
import codecs

def pickle_object(i_obj, i_output_file_path=None):
    pickled_obj_str = codecs.encode(pickle.dumps(i_obj), "base64").decode()
    
    if i_output_file_path != None:
        with open(i_output_file_path, "w") as f:
            f.write(pickled_obj_str)

    return pickled_obj_str

def unpickle_object_from_str(i_pickle_obj_str):
    return pickle.loads(codecs.decode(i_pickle_obj_str.encode(), "base64"))

def unpickle_object_from_file(i_pickle_file_fp):
    with open(i_pickle_file_fp, "r") as f:
        return unpickle_object_from_str(f.read())
##########################################

def random_crop(x):
    return random_crop_model.predict(x)


def get_lr_schedule_func(initial_lr, lr_decay):

    def lr_schedule_func(epoch):
        return initial_lr * (lr_decay ** epoch)

    return lr_schedule_func


def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param


def main(args):

    # parameters
    config = yaml_load(args.config)
    data_name = config['data']['name']
    input_shape = [int(i) for i in config['data']['input_shape']]
    num_classes = int(config['data']['num_classes'])
    num_filters = config['model']['filters']
    kernel_sizes = config['model']['kernels']
    strides = config['model']['strides']
    l1p = float(config['model']['l1'])
    l2p = float(config['model']['l2'])
    skip = bool(config['model']['skip'])
    avg_pooling = bool(config['model']['avg_pooling'])
    batch_size = config['fit']['batch_size']
    num_epochs = config['fit']['epochs']
    verbose = config['fit']['verbose']
    patience = config['fit']['patience']
    save_dir = config['save_dir']
    model_name = config['model']['name']
    loss = config['fit']['compile']['loss']
    model_file_path = os.path.join(save_dir, 'model_best.h5')

    # quantization parameters
    if 'quantized' in model_name:
        logit_total_bits = config["quantization"]["logit_total_bits"]
        logit_int_bits = config["quantization"]["logit_int_bits"]
        activation_total_bits = config["quantization"]["activation_total_bits"]
        activation_int_bits = config["quantization"]["activation_int_bits"]
        alpha = config["quantization"]["alpha"]
        use_stochastic_rounding = config["quantization"]["use_stochastic_rounding"]
        logit_quantizer = config["quantization"]["logit_quantizer"]
        activation_quantizer = config["quantization"]["activation_quantizer"]
        final_activation = bool(config['model']['final_activation'])

    # optimizer
    optimizer = getattr(tf.keras.optimizers, config['fit']['compile']['optimizer'])
    initial_lr = config['fit']['compile']['initial_lr']
    lr_decay = config['fit']['compile']['lr_decay']

    kwargs = {'input_shape': input_shape,
              'num_classes': num_classes,
              'num_filters': num_filters,
              'kernel_sizes': kernel_sizes,
              'strides': strides,
              'l1p': l1p,
              'l2p': l2p,
              'skip': skip,
              'avg_pooling': avg_pooling}

    # pass quantization params
    if 'quantized' in model_name:
        kwargs["logit_total_bits"] = logit_total_bits
        kwargs["logit_int_bits"] = logit_int_bits
        kwargs["activation_total_bits"] = activation_total_bits
        kwargs["activation_int_bits"] = activation_int_bits
        kwargs["alpha"] = None if alpha == 'None' else alpha
        kwargs["use_stochastic_rounding"] = use_stochastic_rounding
        kwargs["logit_quantizer"] = logit_quantizer
        kwargs["activation_quantizer"] = activation_quantizer
        kwargs["final_activation"] = final_activation

    # define model
    model = getattr(resnet_v1_eembc, model_name)(**kwargs)

    # print model summary
    print('#################')
    print('# MODEL SUMMARY #')
    print('#################')
    print(model.summary())
    print('#################')


    # compile model with optimizer
    model.compile(optimizer=optimizer(learning_rate=initial_lr),
                  loss=loss,
                  metrics=['accuracy'])

    # restore "best" model
    model.load_weights(args.pretrained_model)
    
    #Extract model parameters
    sfi_model = StatFI(model)
    params_and_quants = sfi_model.get_params_and_quantizers()
    print(f"{params_and_quants[0].shape}")
    print(f"{params_and_quants[1].shape}")

    fp_pickle = os.path.join(save_dir, model_name) 
    pickle_object(params_and_quants, f"{fp_pickle}_params_and_quantizers.pkl")
    print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="baseline.yml", help="specify yaml config")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="specify pretrained model file path",
    )

    args = parser.parse_args()

    main(args)