import os

import argparse
import datetime
import pathlib

from torch.utils.data import ConcatDataset

from source.dataset_collection import DeepVIDv2Dataset
from source.network_collection import DeepVIDv2
from source.worker_collection import DeepVIDv2Worker
from source.utils import JsonSaver


def get_parser(parent=None):
    if parent is None:
        parser = argparse.ArgumentParser()
    else:
        parser = parent.add_parser("train", help="train mode")

    parser.set_defaults(mode="train")
    # fmt: off
    # dataset params
    dataset_group = parser.add_argument_group("dataset", "dataset parameters")
    dataset_group.add_argument("--noisy-data-paths", "--noisy", type=str, default=["./datasets", ], nargs="+", help="path to noisy data")
    dataset_group.add_argument("-n", "--input-pre-post-frame", "--input", type=int, default=3, help="# of input frames (N)")
    dataset_group.add_argument("-m", "--edge-pre-post-frame", "--edge", type=int, default=None, help="# of edge frames (M)")

    dataset_group.add_argument("--pre-post-omission", type=int, default=0, help="# of omitted frames")
    dataset_group.add_argument("--blind-pixel-ratio", type=float, default=0.005, help="ratio of blind pixels")
    dataset_group.add_argument("--blind-pixel-method", type=str, default="replace", help="method of blind pixels")
    dataset_group.add_argument("--edge-gaussian-size", type=int, default=9, help="size of the gaussian kernel for edge extraction")
    dataset_group.add_argument("--edge-gaussian-sigma", type=float, default=1.0, help="sigma of the gaussian kernel for edge extraction")
    dataset_group.add_argument("--edge-source-frame", type=str, default="mean", help="source frame for edge extraction")

    # network params
    network_group = parser.add_argument_group("network", "network parameters")
    network_group.add_argument("--kernel-size", type=int, default=3, help="size of the kernel")
    network_group.add_argument("--stride", type=int, default=2, help="size of the stride")
    network_group.add_argument("--padding", type=int, default=1, help="size of the padding")
    network_group.add_argument("--out-channels", type=int, default=1, help="# of output channels")
    network_group.add_argument("--num-feature", type=int, default=64, help="# of features")
    network_group.add_argument("--num-blocks", type=int, default=4, help="# of blocks")
    network_group.add_argument("--norm-type", type=str, default="batch", help="type of normalization")
    network_group.add_argument("--activation-type", type=str, default="prelu", help="type of activation")
    network_group.add_argument("--resblock-activation-out-type", type=str, default=None, help="type of resblock activation")

    # training worker params
    worker_group = parser.add_argument_group("worker", "worker parameters")
    # worker params
    worker_group.add_argument("--model-string", "--model", type=str, default=None, help="model string")
    worker_group.add_argument("--num-times-through-data", "--epochs", type=int, default=3, help="# of times through data")
    worker_group.add_argument("--batch-per-step", type=int, default=None, help="# of batches per step")
    # dataset params
    worker_group.add_argument("--batch-size", type=int, default=4, help="size of the batch")
    worker_group.add_argument("--num-workers", type=int, default=0, help="# of workers")
    worker_group.add_argument("--num-sample-val", type=int, default=1000, help="# of validation samples")
    worker_group.add_argument("--sampler-val-type", type=str, default="SubsetRandomSampler", help="type of sampler")
    # scaler params
    worker_group.add_argument("--no-amp", action="store_true", help="disable automatic mixed precision")
    # loss params
    worker_group.add_argument("--loss-type", type=str, default="mse_with_mask", help="type of loss")
    worker_group.add_argument("--l1-reg", type=float, default=1e-2, help="L1 regularization")
    worker_group.add_argument("--l2-reg", type=float, default=1e-2, help="L2 regularization")
    # optimizer params
    worker_group.add_argument("--learning-rate", type=float, default=None, help="learning rate")
    # scheduler params
    worker_group.add_argument("--scheduler-type", type=str, default="ReduceLROnPlateau", help="type of scheduler")
    # callback params
    worker_group.add_argument("--step-save-model", type=int, default=500, help="step to save model")
    worker_group.add_argument("--step-save-image", type=int, default=100, help="step to save image")

    # fmt: on
    if parent is None:
        return parser
    else:
        return parent


def process_args(args):
    # if noisy_data is folder, load all files in the folder
    if os.path.isdir(args.noisy_data_paths[0]):
        args.is_folder = True
        root = args.noisy_data_paths[0]
        filenames = os.listdir(root)
        filenames.sort()
        args.noisy_data_paths = [os.path.join(root, filename) for filename in filenames]
    else:
        args.is_folder = False

    # init params if not set
    if args.learning_rate is None:
        if args.is_folder:
            args.learning_rate = 5e-6
        else:
            args.learning_rate = 1e-4

    if args.batch_per_step is None:
        if args.is_folder:
            args.batch_per_step = 360
        else:
            args.batch_per_step = 12

    # set params
    args.in_channels = args.input_pre_post_frame * 2 + 1 + 4  # number of input channels
    if args.model_string is None:
        run_uid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.model_string = run_uid
    args.output_dir = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "results",
        args.model_string,
    )

    return args


def run_worker(args):
    # create dataset
    print("Creating dataset...")
    args_local = argparse.Namespace(**vars(args))
    dataset_list = []
    for noisy_data_path in args.noisy_data_paths:
        args_local.noisy_data_path = noisy_data_path
        dataset_local = DeepVIDv2Dataset(args_local)
        dataset_list.append(dataset_local)
    dataset = ConcatDataset(dataset_list)

    # create network
    print("Creating network...")
    network = DeepVIDv2(args)

    # create worker
    print("Creating worker...")
    trainer = DeepVIDv2Worker(dataset, network, args)

    # run worker
    print("Running worker...")
    trainer.run()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    args = process_args(args)

    # make output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "config"), exist_ok=True)

    # save config
    sorted_args = dict(sorted(vars(args).items()))
    path_config = os.path.join(args.output_dir, "config", "config_train.json")
    json_obj = JsonSaver(sorted_args)
    json_obj.save_json(path_config)

    # run worker
    run_worker(args)
