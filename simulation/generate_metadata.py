import argparse
import os
import json
import pandas as pd
import numpy as np

def main(args):
    bounds = []
    for i in range(min(len(args.lower_bounds), len(args.upper_bounds))):
        bounds.append([args.lower_bounds[i], args.upper_bounds[i]])
    dim = len(args.cartesian_idx)
    data_dim = None
    vel_list, acc_list = [], []
    # Find files in dataset directory
    file_names = [file for file in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, file))]
    for idx, file_name in enumerate(file_names):
        if 'particles' not in file_name:
            continue
        # Load data and reshape it
        df_data = pd.read_csv(os.path.join(args.data_dir, file_name), header=None)
        data_dim = df_data.shape[1]
        data = np.array(df_data).reshape(args.timesteps, -1, data_dim)
        # Calculate velocities and accelerations of each node for each timestep
        velocities    = np.diff(data[:,:,args.cartesian_idx], n=1, axis=0)
        accelerations = np.diff(velocities, n=1, axis=0)
        # print(velocities.shape, accelerations.shape)
        vel_list.append(velocities.reshape(-1, dim))
        acc_list.append(accelerations.reshape(-1, dim))
    # Put metadata into dict
    vel_stack = np.concatenate(vel_list, axis=0)
    acc_stack = np.concatenate(acc_list, axis=0)
    # print(vel_stack.shape, acc_stack.shape)
    metadata = {
        'cartesian_idx': args.cartesian_idx,
        'control_idx': args.control_idx,
        'material_id': args.material_id,
        'bounds': bounds,
        'sequence_length': args.timesteps,
        'dim': dim,
        'data_dim': data_dim,
        'vel_mean': list(np.mean(vel_stack, axis=0)),
        'vel_std': list(np.std(vel_stack, axis=0)),
        'acc_mean': list(np.mean(acc_stack, axis=0)),
        'acc_std': list(np.std(acc_stack, axis=0))
    }
    # Save metadata on target dir or if not defined to the data directory
    target_file = os.path.join(args.target_dir, 'metadata.json') if args.target_dir is not None else os.path.join(args.data_dir, 'metadata.json')
    with open(target_file, 'w') as fp:
        json.dump(metadata, fp)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Postprocess csv files.')
    # Define datapaths
    parser.add_argument('-d', '--data_dir', default='/home/mulerod1/projects/mpm-simulations/deps/taichi/outputs/mpm/p1e4/frames',
                        help='Dataset directory', required=True)
    parser.add_argument('--target_dir', default=None, help='Directory where processed data files will be saved')
    parser.add_argument('-t', '--timesteps', type=int, default=400, help='How many timesteps does the simulations have')
    parser.add_argument('--upper_bounds', nargs='+', type=float, help="Upper boundaries of the simulation", required=True)
    parser.add_argument('--lower_bounds', nargs='+', type=float, help="Lower boundaries of the simulation", required=True)
    parser.add_argument('--cartesian_idx', nargs='+', type=int, help="Cartesian indices in the data", required=True)
    parser.add_argument('--control_idx', nargs='+', type=int, help="Control input indices in the data", required=True)
    parser.add_argument('--material_id', type=int, help="Material index in the data", required=True)
    args = parser.parse_args()
    main(args)