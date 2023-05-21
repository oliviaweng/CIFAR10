import os
import time

if os.system("nvidia-smi") == 0:
    import setGPU
import tensorflow as tf
import glob
import sys
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
import resnet_v1_eembc
import yaml
import csv
from fkeras.fmodel import FModel
from fkeras.metrics.hessian import HessianMetrics

# ICCAD2023 related imports ##############
import iccad_2023_experiment_utils as ieu
import subprocess
##########################################

# from keras_flops import get_flops # (different flop calculation)
# import kerop
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop
from tensorflow.keras.losses import CategoricalCrossentropy

random_crop_model = tf.keras.models.Sequential()
random_crop_model.add(
    RandomCrop(
        32,
        32,
        input_shape=(
            32,
            32,
            3,
        ),
    )
)


def random_crop(x):
    return random_crop_model.predict(x)


def get_lr_schedule_func(initial_lr, lr_decay):
    def lr_schedule_func(epoch):
        return initial_lr * (lr_decay**epoch)

    return lr_schedule_func


def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param


def pre_exit_procedure(open_files):
    print("[pre_exit_procedure] Manual exit initiated. Closing open experiment files.")
    for f in open_files:
        f.write("[pre_exit_procedure] Manual exit initiated. Closing this file.")
        f.close()
    exit()


def exp_file_write(file_path, input_str, open_mode="a"):
    with open(file_path, open_mode) as f:
        f.write(input_str)


def main(args):
    # S: Running eagerly is essential. Without eager execution mode,
    ### the fkeras.utils functions (e.g., gen_mask_tensor) only get
    ### get evaluated once and then subsequent "calls" reuse the
    ### same value from the initial call (which manifest as the
    ### same fault(s) being injected over and over again)
    tf.config.run_functions_eagerly(True)
    # tf.data.experimental.enable_debug_mode()

    efd_fp = args.efd_fp  # "./efd_val_inputs_0-31_with_eager_exec_cleaning.log"
    efr_fp = args.efr_fp  # "./efr_val_inputs_0-31_with_eager_exec_cleaning.log"
    if args.efx_overwrite:
        exp_file_write(efd_fp, "", "w")
        exp_file_write(efr_fp, "", "w")
    print(args)

    # parameters
    config = yaml_load(args.config)
    data_name = config["data"]["name"]
    input_shape = [int(i) for i in config["data"]["input_shape"]]
    num_classes = int(config["data"]["num_classes"])
    num_filters = config["model"]["filters"]
    kernel_sizes = config["model"]["kernels"]
    strides = config["model"]["strides"]
    l1p = float(config["model"]["l1"])
    l2p = float(config["model"]["l2"])
    skip = bool(config["model"]["skip"])
    avg_pooling = bool(config["model"]["avg_pooling"])
    # batch_size = config["fit"]["batch_size"]
    num_epochs = config["fit"]["epochs"]
    verbose = config["fit"]["verbose"]
    patience = config["fit"]["patience"]
    save_dir = config["save_dir"]
    model_name = config["model"]["name"]
    loss = config["fit"]["compile"]["loss"]
    batch_size = args.batch_size
    model_file_path = args.pretrained_model

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # quantization parameters
    if "quantized" in model_name:
        logit_total_bits = config["quantization"]["logit_total_bits"]
        logit_int_bits = config["quantization"]["logit_int_bits"]
        activation_total_bits = config["quantization"]["activation_total_bits"]
        activation_int_bits = config["quantization"]["activation_int_bits"]
        alpha = config["quantization"]["alpha"]
        use_stochastic_rounding = config["quantization"]["use_stochastic_rounding"]
        logit_quantizer = config["quantization"]["logit_quantizer"]
        activation_quantizer = config["quantization"]["activation_quantizer"]
        final_activation = bool(config["model"]["final_activation"])

    # optimizer
    optimizer = getattr(tf.keras.optimizers, config["fit"]["compile"]["optimizer"])
    initial_lr = config["fit"]["compile"]["initial_lr"]
    lr_decay = config["fit"]["compile"]["lr_decay"]

    # load dataset
    if data_name == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train, X_test = X_train / 256.0, X_test / 256.0

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    elif data_name == "particlezoo":
        import particlezoo

        (X_train, y_train), (X_test, y_test) = particlezoo.load_data()
        X_train, X_test = X_train / 256.0, X_test / 256.0

    if loss == "squared_hinge":
        y_train = y_train * 2 - 1  # -1 or 1 for hinge loss
        y_test = y_test * 2 - 1

    # define data generator
    if data_name == "cifar10":
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            # preprocessing_function=random_crop,
            # brightness_range=(0.9, 1.2),
            # contrast_range=(0.9, 1.2)
        )
    elif data_name == "particlezoo":
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

    kwargs = {
        "input_shape": input_shape,
        "num_classes": num_classes,
        "num_filters": num_filters,
        "kernel_sizes": kernel_sizes,
        "strides": strides,
        "l1p": l1p,
        "l2p": l2p,
        "skip": skip,
        "avg_pooling": avg_pooling,
    }

    # pass quantization params
    if "quantized" in model_name:
        kwargs["logit_total_bits"] = logit_total_bits
        kwargs["logit_int_bits"] = logit_int_bits
        kwargs["activation_total_bits"] = activation_total_bits
        kwargs["activation_int_bits"] = activation_int_bits
        kwargs["alpha"] = None if alpha == "None" else alpha
        kwargs["use_stochastic_rounding"] = use_stochastic_rounding
        kwargs["logit_quantizer"] = logit_quantizer
        kwargs["activation_quantizer"] = activation_quantizer
        kwargs["final_activation"] = final_activation

    # define model
    model = getattr(resnet_v1_eembc, model_name)(**kwargs)

    # print model summary
    print("#################")
    print("# MODEL SUMMARY #")
    print("#################")
    print(model.summary())
    print("#################")

    # compile model with optimizer
    model.compile(
        optimizer=optimizer(learning_rate=initial_lr),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # restore "best" model
    model.load_weights(model_file_path)

    # S: Instantiate the FKeras model to be used
    fmodel = FModel(model, 0.0)
    print(fmodel.layer_bit_ranges)

    # S: Configure how many validation inputs will be used
    curr_val_input = X_test
    curr_val_output = y_test
    if 0 < args.num_val_inputs <= X_test.shape[0]:
        curr_val_input = X_test[: args.num_val_inputs]
        curr_val_output = y_test[: args.num_val_inputs]
    else:
        raise RuntimeError("Improper configuration for 'num_val_inputs'")

    # S: Configure which bits will be flipped
    bit_flip_range_step = (0, 2, 1)
    bit_flip_range_step = (0, fmodel.num_model_param_bits, 1)
    if args.use_custom_bfr == 1:
        bfr_start_ok = (0 <= args.bfr_start) and (
            args.bfr_start <= fmodel.num_model_param_bits
        )
        bfr_end_ok = (0 <= args.bfr_end) and (
            args.bfr_end <= fmodel.num_model_param_bits
        )
        bfr_ok = bfr_start_ok and bfr_end_ok
        if bfr_ok:
            bit_flip_range_step = (args.bfr_start, args.bfr_end, args.bfr_step)
        else:
            raise RuntimeError("Improper configuration for bit flipping range")
        
    if args.correct_idx_file is None:
        # Get non-faulty model predictions
        non_faulty_preds = model.predict(X_test, batch_size=batch_size)
        non_faulty_preds = tf.one_hot(tf.argmax(non_faulty_preds, axis=1), depth=10)
        non_faulty_preds = tf.reshape(non_faulty_preds, y_test.shape)

        # one-hot accuracy
        acc = compute_accuracy(y_test, non_faulty_preds)
        print("non faulty accuracy = ", acc.numpy())

        # Pull out the correct predictions
        non_faulty_correct_indices = tf.where(
            tf.reduce_all(tf.equal(non_faulty_preds, y_test), axis=1)
        )
        print("Saving correct indices to file")
        np.save(os.path.join(save_dir, "non_faulty_correct_indices.npy"), non_faulty_correct_indices.numpy())
    else:
        print("Loading correct indices from file")
        non_faulty_correct_indices = np.load(args.correct_idx_file)
    X_test = X_test[non_faulty_correct_indices]
    X_test = tf.reshape(X_test, (X_test.shape[0], 32, 32, 3))
    y_test = y_test[non_faulty_correct_indices]
    y_test = tf.reshape(y_test, (y_test.shape[0], 10))

    # S: Begin the single fault injection (bit flipping) campaign
    for bit_i in range(*bit_flip_range_step):

        # S: Flip the desired bit in the model
        fmodel.explicit_select_model_param_bitflip([bit_i])

        # get predictions
        time_start = time.time()
        y_pred = model.predict(X_test, batch_size=batch_size)
        loss_val = CategoricalCrossentropy()(y_test, y_pred)
        predict_time = time.time() - time_start
        print(f"Time to predict = {predict_time}")

        # one-hot encode
        y_pred = tf.one_hot(tf.argmax(y_pred, axis=1), depth=10)
        y_pred = tf.reshape(y_pred, y_test.shape)

        y_pred_correct_indices = tf.where(
            tf.reduce_all(tf.equal(y_pred, y_test), axis=1)
        )
        # Number of times bit flip caused a misprediction
        # @Andy: Log this number for each bit flip
        num_mispredictions = y_test.shape[0] - len(y_pred_correct_indices)
        print(f"num mispredictions = {y_test.shape[0] - len(y_pred_correct_indices)}")

        print("cross entropy loss = %.3f" % loss_val)

        #S: Compute Hessian
        hess_start = time.time()
        hess = HessianMetrics(
            fmodel.model,
            CategoricalCrossentropy(),
            curr_val_input,
            curr_val_output,
            batch_size=batch_size,
        )
        hess_trace = hess.trace(tolerance=1e-1)
        trace_time = time.time() - hess_start
        print(f"Hessian trace compute time: {trace_time} seconds")
        print(f"hess_trace = {hess_trace}")

        #S: Specify pefx file paths
        fp_pefr = os.path.join(args.ieu_efx_dir, args.ieu_model_id, args.ieu_pefr_name)
        fp_pefd = os.path.join(args.ieu_efx_dir, args.ieu_model_id, args.ieu_pefd_name)

        #S: Update corresponding pefr file
        time_store_pefr = ieu.store_pefr_cifar10(fp_pefr, bit_i, num_mispredictions, hess_trace, loss_val)

        #S: Update corresponding pefd file
        subtime_dataset   = 0
        subtime_gt_metric = predict_time
        subtime_ht_metric = trace_time
        subtime_pefr      = time_store_pefr
        my_sub_times = (subtime_dataset, subtime_gt_metric, subtime_ht_metric, subtime_pefr)
        time_store_pefd = ieu.store_pefd_experiment1(fp_pefd, bit_i, time.time()-10, my_sub_times)

        # #S: Update IEU FKeras-Experiments repo by pushing pefd files (pefr files not tracked until later)
        # bits_flipped_by_vsystem = args.bfr_start - args.ieu_lbi
        # if bits_flipped_by_vsystem%args.ieu_git_step == 0: 
        #     subprocess.run("./scripts/iccad_2023_experiment1_git_commands.sh", shell=True)
        
        break



def compute_accuracy(one_hot_true, one_hot_pred):
    # Convert one-hot encoding to indices
    true_indices = tf.argmax(one_hot_true, axis=1)
    pred_indices = tf.argmax(one_hot_pred, axis=1)

    # Compare predicted indices with true indices
    correct_predictions = tf.equal(true_indices, pred_indices)

    # Compute accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="baseline.yml", help="specify yaml config"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="specify pretrained model file path",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="specify batch size")
    # I: Arguments for bit flipping experiment
    parser.add_argument(
        "--efd_fp",
        type=str,
        default="./efd.log",
        help="File path for experiment file with debugging data",
    )
    parser.add_argument(
        "--efr_fp",
        type=str,
        default="./efr.log",
        help="File path for experiment file with result data",
    )
    parser.add_argument(
        "--efx_overwrite",
        type=int,
        default=0,
        help="If '0', efd_fp and efr_fp are appended to with data; If '1', efd_fp and efr_fp are overwritten with data",
    )
    parser.add_argument(
        "--use_custom_bfr",
        type=int,
        default=0,
        help="If '0', all bits (of supported layers) will be flipped. If '1', all bits in the range (--bfr_start, --bfr_end, --bfr_step) will be flipped",
    )
    parser.add_argument(
        "--bfr_start",
        type=int,
        default=0,
        help="Bit flipping range start. Note: bit index starts at 0.",
    )
    parser.add_argument(
        "--bfr_end",
        type=int,
        default=2,
        help="Bit flipping range end (exclusive). Note: bit index starts at 0.",
    )
    parser.add_argument(
        "--bfr_step",
        type=int,
        default=1,
        help="Bit flipping range step size.",
    )
    parser.add_argument(
        "--num_val_inputs",
        type=int,
        default=2,
        help="Number of validation inputs to use for evaluating the faulty models",
    )
    parser.add_argument(
        "--correct_idx_file",
        type=str,
        default=None,
        help="File path to numpy array containing indices of correct predictions",
    )


    parser.add_argument(
        "--ieu_model_id",
        type=str,
        default=None,
        help="IEU identifying string of model.",
    )
    parser.add_argument(
        "--ieu_vinputs",
        type=int,
        default=256,
        help="IEU val inputs for run",
    )
    parser.add_argument(
        "--ieu_vsystem_id",
        type=str,
        default=None,
        help="IEU virtual system id",
    )
    parser.add_argument(
        "--ieu_efx_dir",
        type=str,
        default=None,
        help="IEU data directory for all models",
    )
    parser.add_argument(
        "--ieu_pefr_name",
        type=str,
        default=None,
        help="IEU name of pickled efr file",
    )
    parser.add_argument(
        "--ieu_pefd_name",
        type=str,
        default=None,
        help="IEU name of pickled efd file",
    )
    parser.add_argument(
        "--ieu_git_step",
        type=int,
        default=100,
        help="IEU frequency with which to update the IEU git repo",
    )
    parser.add_argument(
        "--ieu_lbi",
        type=int,
        default=100,
        help="IEU lbi (i.e., start of FI loop)",
    )

    args = parser.parse_args()

    main(args)