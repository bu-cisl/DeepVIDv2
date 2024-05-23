import os

import argparse
import pathlib

from source.dataset_collection import DeepVIDv2Dataset
from source.network_collection import DeepVIDv2
from source.worker_collection import DeepVIDv2Worker
from source.utils import JsonLoader, JsonSaver


def get_parser(parent=None):
    if parent is None:
        parser = argparse.ArgumentParser()
    else:
        parser = parent.add_parser("inference", help="inference mode")

    parser.set_defaults(mode="inference")
    # fmt: off
    # dataset params
    dataset_group = parser.add_argument_group("dataset", "dataset parameters")
    dataset_group.add_argument("--noisy-data-paths", "--noisy", type=str, default=["./datasets", ], nargs="+", help="path to noisy data")

    # inference worker params
    worker_group = parser.add_argument_group("worker", "worker parameters")
    # worker params
    worker_group.add_argument("--model-string", "--model", type=str, default=None, required=True, help="model string")
    # dataset params
    worker_group.add_argument("--batch-size", type=int, default=4, help="size of the batch")
    worker_group.add_argument("--num-workers", type=int, default=0, help="# of workers")
    # scaler params
    worker_group.add_argument("--no-amp", action="store_true", help="disable automatic mixed precision")
    # inference params
    worker_group.add_argument("--model-path", type=str, default=None, help="path to the model")
    worker_group.add_argument("--output-file", type=str, default=None, help="filename of the output file")
    worker_group.add_argument("--output-dtype", type=str, default="float32", help="dtype of the output file")

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

    # set params
    args.output_dir = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "results",
        args.model_string,
    )

    # set default model path
    if args.model_path is None:
        args.model_path = os.path.join(
            args.output_dir,
            "checkpoint",
            "checkpoint_final.ckpt",
        )

    # load training config
    args_training = load_config(
        os.path.join(args.output_dir, "config", "config_train.json")
    )

    # combine args_training and args from command line
    args_combined = argparse.Namespace(**vars(args_training))
    args_combined.__dict__.update(vars(args))

    return args_combined


def load_config(path_config):
    json_obj = JsonLoader(path_config)
    json_obj.load_json()
    args = argparse.Namespace(**json_obj.json_data)
    return args


def run_worker(args):
    args_local = argparse.Namespace(**vars(args))

    for noisy_data_path in args.noisy_data_paths:
        # create dataset
        print("Creating dataset...")
        args_local.noisy_data_path = noisy_data_path

        # set output filename
        if args.output_file is None:
            root, _ = os.path.splitext(noisy_data_path)
            input_filename_no_ext = os.path.basename(root)
            output_filename = "_".join(
                (
                    args.model_string,
                    "result",
                    input_filename_no_ext + ".tiff",
                )
            )
            output_file = os.path.join(
                args.output_dir,
                output_filename,
            )

            args_local.output_file = output_file

        dataset = DeepVIDv2Dataset(args_local)

        # create network
        print("Creating network...")
        network = DeepVIDv2(args_local)

        # create worker
        print("Creating worker...")
        trainer = DeepVIDv2Worker(dataset, network, args_local)

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
    path_config = os.path.join(args.output_dir, "config", "config_inference.json")
    json_obj = JsonSaver(sorted_args)
    json_obj.save_json(path_config)

    # run worker
    run_worker(args)
