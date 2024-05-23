import os

import argparse

from source.utils import JsonSaver
import scripts.train as train
import scripts.inference as inference


def get_parser():
    parser = argparse.ArgumentParser()
    # add subparsers
    subparsers = parser.add_subparsers(required=True)
    subparsers = train.get_parser(subparsers)
    subparsers = inference.get_parser(subparsers)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.mode == "train":
        args = train.process_args(args)
    elif args.mode == "inference":
        args = inference.process_args(args)
    else:
        raise ValueError("Invalid mode")

    # make output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "config"), exist_ok=True)

    # save config
    sorted_args = dict(sorted(vars(args).items()))
    path_config = os.path.join(
        args.output_dir, "config", "config_{}.json".format(args.mode)
    )
    json_obj = JsonSaver(sorted_args)
    json_obj.save_json(path_config)

    # run worker
    if args.mode == "train":
        train.run_worker(args)
    elif args.mode == "inference":
        inference.run_worker(args)
    else:
        raise ValueError("Invalid mode")
