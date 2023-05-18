import os
import time
if os.system("nvidia-smi") == 0:
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
from fkeras.fmodel import FModel
from fkeras.metrics.hessian import HessianMetrics

# from keras_flops import get_flops # (different flop calculation)
# import kerop
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_crossentropy

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
    #S: Running eagerly is essential. Without eager execution mode,
    ### the fkeras.utils functions (e.g., gen_mask_tensor) only get
    ### get evaluated once and then subsequent "calls" reuse the 
    ### same value from the initial call (which manifest as the
    ### same fault(s) being injected over and over again)
    tf.config.run_functions_eagerly(True)
    # tf.data.experimental.enable_debug_mode()

    efd_fp = args.efd_fp #"./efd_val_inputs_0-31_with_eager_exec_cleaning.log"
    efr_fp = args.efr_fp #"./efr_val_inputs_0-31_with_eager_exec_cleaning.log"
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
        optimizer=optimizer(learning_rate=initial_lr), loss=CategoricalCrossentropy(), metrics=["accuracy"]
    )

    # restore "best" model
    model.load_weights(model_file_path)

    #S: Instantiate the FKeras model to be used
    fmodel = FModel(model, 0.0)
    print(fmodel.layer_bit_ranges)

    #S: Configure how many validation inputs will be used
    curr_val_input = X_test
    curr_val_output = y_test
    if 0 < args.num_val_inputs <= X_test.shape[0]:
        curr_val_input = X_test[:args.num_val_inputs]
        curr_val_output = y_test[:args.num_val_inputs]
    else:
        raise RuntimeError("Improper configuration for 'num_val_inputs'")

    #S: Configure which bits will be flipped
    bit_flip_range_step = (0,2, 1)
    bit_flip_range_step = (0,fmodel.num_model_param_bits, 1)
    if (args.use_custom_bfr == 1): 
        bfr_start_ok = (0 <= args.bfr_start) and (args.bfr_start<= fmodel.num_model_param_bits)
        bfr_end_ok   = (0 <= args.bfr_end  ) and (args.bfr_end  <= fmodel.num_model_param_bits)
        bfr_ok = bfr_start_ok and bfr_end_ok
        if bfr_ok:
            bit_flip_range_step = (args.bfr_start, args.bfr_end, args.bfr_step)
        else:
            raise RuntimeError("Improper configuration for bit flipping range")

    #S: Begin the single fault injection (bit flipping) campaign
    for bit_i in range(*bit_flip_range_step):

        #S: Flip the desired bit in the model 
        fmodel.explicit_select_model_param_bitflip([bit_i])

        # get predictions
        y_pred = model.predict(curr_val_input, batch_size=batch_size)
        loss_val = CategoricalCrossentropy()(curr_val_output, y_pred)

        print("cross entropy loss = %.3f" % loss_val)

        hess_start = time.time()
        hess = HessianMetrics(
            fmodel.model, 
            CategoricalCrossentropy(), 
            curr_val_input, 
            curr_val_output, 
            batch_size=batch_size,
        )
        hess_trace = hess.trace(max_iter=500)
        trace_time = time.time() - hess_start
        print(f"Hessian trace compute time: {trace_time} seconds")
        print(f"hess_trace = {hess_trace}")
        exp_file_write(
            os.path.join(save_dir, "hess_trace_debug.log"), 
            f"num_val_inputs = {args.num_val_inputs} | batch_size = {batch_size}\n"
        )
        exp_file_write(os.path.join(save_dir, "hess_trace_debug.log"), f"Time = {trace_time} seconds\n")
        exp_file_write(os.path.join(save_dir, "hess_trace_debug.log"), f"Trace = {hess_trace}\n")
        # exp_file_write(efd_fp, f'Hessian trace compute time: {time.time() - hess_start} seconds\n')
        # exp_file_write(efd_fp,  f"hess_trace = {hess_trace}\n")
        # break


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
    parser.add_argument(
        "--batch_size", type=int, default=32, help="specify batch size"
    )
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

    args = parser.parse_args()

    main(args)
