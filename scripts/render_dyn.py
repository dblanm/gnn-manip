# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#
# import sys
import numpy as np
import subprocess
import argparse
import os

from gnn_manip.utils.rollout_utils import get_dataset, get_model, compute_rollout, create_arg_parser


def get_rollout(args):
    # Get dataset
    dataset = get_dataset(args)
    nof_steps = dataset.time_steps - dataset.k
    print("Data loaded")

    # Get and load the model
    gnn_model = get_model(dataset, args)

    # Generate rollout
    # start_time = time.time()
    prediction = compute_rollout(dataset, gnn_model, args)
    # rollout_time = time.time() - start_time
    # print(f'Rollout generation takes {rollout_time:.2f} seconds')

    return prediction, nof_steps


def render_blender(file_name, end=300, args=None):
    arguments = [
                "blender", '--background', '--python', args.blender_file, '--',
                '--file_name', file_name, '--output', args.output,
                '--start', str(1), '--end', str(end), '--step', str(args.step),
                '--res', str(args.res), '--sand_color', str(args.sand_color),
                '--diameter', str(args.diameter), '--camera_idx', str(args.camera_idx)
            ]
    if args.use_transparent_background:
        arguments.append('--use_transparent_background')
    if args.hide_rigids:
        arguments.append('--hide_rigids')
    if args.hide_background_objects:
        arguments.append('--hide_background_objects')
    if args.save_ffmpeg:
        arguments.append('--save_ffmpeg')
    if args.camera_position is not None:
        arguments.append('--camera_position')
        for pos in args.camera_position:
            arguments.append(str(pos))
    subprocess.run(arguments)


def save_as_csv(prediction_npy, args):
    csv_file_path = os.path.join(args.output, 'positions.csv')
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # Read .npy and take only relevant data
    data = prediction_npy[:, :, :5]
    # data = data[args.start:args.end,:,:]
    # print(data.shape)
    data_dim = data.shape[-1]
    data = np.reshape(data, (-1, data_dim))
    # Save .csv
    format = '%d,%d,%f,%f,%f' if data_dim == 5 else '%f'
    np.savetxt(csv_file_path, data, delimiter=",", fmt=format)
    return csv_file_path


def main(args):
    prediction_npy, nof_steps = get_rollout(args)
    csv_file_path = save_as_csv(prediction_npy, args)
    # render_blender(csv_file_path, nof_steps, args=args)
    return


if __name__ == '__main__':
    args = create_arg_parser()
    main(args)
