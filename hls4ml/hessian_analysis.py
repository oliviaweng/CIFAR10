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

import codecs
import pickle
from tensorflow.keras.losses import CategoricalCrossentropy
from fkeras.metrics.hessian import HessianMetrics
import time

import numpy as np

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

    # load dataset
    if data_name == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train, X_test = X_train/256., X_test/256.

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    elif data_name == 'particlezoo':
        import particlezoo
        (X_train, y_train), (X_test, y_test) = particlezoo.load_data()
        X_train, X_test = X_train/256., X_test/256.

    if loss == 'squared_hinge':
        y_train = y_train * 2 - 1  # -1 or 1 for hinge loss
        y_test = y_test * 2 - 1

    # define data generator
    if data_name == 'cifar10':
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            # preprocessing_function=random_crop,
            # brightness_range=(0.9, 1.2),
            # contrast_range=(0.9, 1.2)
        )
    elif data_name == 'particlezoo':
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            # vertical_flip=True,
            # zca_whitening=True,
            # brightness_range=(0.9, 1.2),
            # contrast_range=(0.9, 1.2)
        )

    # run preprocessing on training dataset
    datagen.fit(X_train)

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

    # analyze FLOPs (see https://github.com/kentaroy47/keras-Opcounter)
    # layer_name, layer_flops, inshape, weights = kerop.profile(model)

    # # visualize FLOPs results
    # total_flop = 0
    # for name, flop, shape in zip(layer_name, layer_flops, inshape):
    #     print("layer:", name, shape, " MFLOPs:", flop/1e6)
    #     total_flop += flop
    # print("Total FLOPs: {} MFLOPs".format(total_flop/1e6))

    # tf.keras.utils.plot_model(model,
    #                           to_file="model.png",
    #                           show_shapes=True,
    #                           show_dtype=False,
    #                           show_layer_names=False,
    #                           rankdir="TB",
    #                           expand_nested=False)

    # Alternative FLOPs calculation (see https://github.com/tokusumi/keras-flops), ~same answer
    # total_flop = get_flops(model, batch_size=1)
    # print("FLOPS: {} GLOPs".format(total_flop/1e9))

    # compile model with optimizer
    model.compile(optimizer=optimizer(learning_rate=initial_lr),
                  loss=loss,
                  metrics=['accuracy'])

    # callbacks
    # from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

    # lr_schedule_func = get_lr_schedule_func(initial_lr, lr_decay)

    # callbacks = [ModelCheckpoint(model_file_path, monitor='val_accuracy', verbose=verbose, save_best_only=True),
    #              EarlyStopping(monitor='val_accuracy', patience=patience, verbose=verbose, restore_best_weights=True),
    #              LearningRateScheduler(lr_schedule_func, verbose=verbose),
    #              ]

    # # train
    # history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
    #                     steps_per_epoch=X_train.shape[0] // batch_size,
    #                     epochs=num_epochs,
    #                     validation_data=(X_test, y_test),
    #                     callbacks=callbacks,
    #                     verbose=verbose)

    # restore "best" model
    model.load_weights(model_file_path)
    
    # Evaluate the model
    print("Computing Hessian Metrics...")
    
    hess = HessianMetrics(
        model, 
        CategoricalCrossentropy(), 
        X_test[:32], 
        y_test[:32],
        batch_size=32
    )
        
    hess_start = time.time()
    top_k = 8
    BIT_WIDTH = 8
    strategy = "sum"
    # Hessian model-wide sensitivity ranking
    eigenvalues, eigenvectors = hess.top_k_eigenvalues(k=top_k, max_iter=500)
    sanitized_evs = list() #the list contains k lists of eigenvectors
    print("Eigenvectors")
    for i in range(len(eigenvalues)):
        print(f"Top {i+1} eigenvalue: {eigenvalues[i]}")
    for i in range(top_k):
        curr_evs = list()
        # print(eigenvectors[i])
        print(f"Eigenvector shape i: {np.array(eigenvectors[i]).shape}")
        for j in range(len(eigenvectors[i])):
            print(f"--Eigenvector shape j: {np.array(eigenvectors[i][j]).shape}")
            if np.array(eigenvectors[i][j]).size > 32:
                curr_evs.append(np.array(eigenvectors[i][j]))
        sanitized_evs.append(np.array(curr_evs))
    
    for i in range(top_k):
        for j in range(len(sanitized_evs[i])):
            print(f" Sanitized i,j {i},{j} = {sanitized_evs[i][j].shape}")
        break

    print(f'Hessian eigenvalue compute time: {time.time() - hess_start} seconds\n')
    # eigenvalues = None
    rank_start_time = time.time()

    param_ranking, param_scores = hess.hessian_ranking_general(
        eigenvectors, eigenvalues=eigenvalues, k=top_k, strategy=strategy
    )
    # bitwise_rank, bitwise_scores = hess.rank_bits(param_scores, 5) # add m = 5 bits (doesn't work; TODO: delete)
    bitwise_rank = hess.convert_param_ranking_to_msb_bit_ranking(param_ranking, BIT_WIDTH)

    pickled_ranking_file = f"hessian_ranked_model_bits_iccad_2023_CIFAR-10.pkl"
    
    obj = list(bitwise_rank)
    pickled_obj = codecs.encode(pickle.dumps(obj), "base64").decode()
    with open(pickled_ranking_file, "w") as f:
        f.write(pickled_obj)


    gradient_rank, _ = hess.layer_gradient_ranking_general()
    bitwise_rank = hess.convert_param_ranking_to_msb_bit_ranking(gradient_rank, BIT_WIDTH)
    
    pickled_ranking_file = f"gradient_ranked_model_bits_iccad_2023_CIFAR-10.pkl"
    obj = list(bitwise_rank)
    pickled_obj = codecs.encode(pickle.dumps(obj), "base64").decode()
    with open(pickled_ranking_file, "w") as f:
        f.write(pickled_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="baseline.yml", help="specify yaml config")

    args = parser.parse_args()

    main(args)
