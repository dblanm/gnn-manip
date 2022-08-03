import argparse
import os
import pandas as pd
import numpy as np

def main(args):
    # Find files in dataset directory
    file_names = [file for file in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, file))]
    file_names.sort()
    sim_data = np.zeros((len(file_names), 2))
    for idx, file_name in enumerate(file_names):
        # Load data and reshape it
        df_data = pd.read_csv(os.path.join(args.data_dir, file_name), header=None)
        data_dim = df_data.shape[1]
        data = np.array(df_data).reshape(args.timesteps, -1, data_dim)
        nof_particles = data.shape[1]
        # Filter container particles out of data
        # (The container particles are rigid body particles that are edded to the scene before sand particles)
        first_sand_id = None
        for i in range(nof_particles):
            if data[0, i, args.material_id] < 0.5:
                first_sand_id = i
                break
        data_filtered = data[:,first_sand_id:,:]
        nof_particles_filtered = data_filtered.shape[1]
        # Filter out too fast particles
        if args.filter_velocities is not None:
            accepted_particle_idx = []
            for i in range(nof_particles_filtered):
                max_speed = np.max(np.sqrt((data_filtered[1:, i, args.cartesian_idx] - data_filtered[:-1, i, args.cartesian_idx])**2))
                if max_speed < args.filter_velocities:
                    accepted_particle_idx.append(i)
            # Reshape data back to original
            data_filtered = data_filtered[:,accepted_particle_idx,:]
            nof_particles_filtered = data_filtered.shape[1]
        #Remove first 100 timesteps
        data_filtered = data_filtered[100:]
        # Remove last 100 timesteps
        # data_filtered = data_filtered[:-100]
        # print(data_filtered.shape)
        # Reshape data
        data_filtered = data_filtered.reshape(-1, data_dim)
        # Save data
        new_file_name = f'particles_{idx+1:06d}.csv'
        np.savetxt(os.path.join(args.target_dir, new_file_name), data_filtered, fmt=args.target_fmt, delimiter=',')
        print(f'{file_name} contains originally {nof_particles} particles and now {nof_particles_filtered} particles.')
        # Update simulation data
        sim_data[idx,:] = np.array([idx+1, nof_particles_filtered])
    # Save simulation data
    np.savetxt(os.path.join(args.target_dir, 'sim_data.csv'), sim_data, fmt='%d', delimiter=',')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Postprocess csv files.')
    # Define datapaths
    parser.add_argument('-d', '--data_dir', help='Dataset directory')
    parser.add_argument('--target_dir', help='Directory where processed data files will be saved')
    parser.add_argument('-t', '--timesteps', type=int, default=400, help='How many timesteps does the simulations have')
    parser.add_argument('--material_id', type=int, default=1, help='Id of column containing material attribute')
    parser.add_argument('--cartesian_idx', nargs='+', type=int, help='Idx of column containing cartesian coordinates')
    parser.add_argument('--filter_velocities', type=float, default=None, help='Filter particles with higher velocity at any frame than FILTER_VELCOCITIES')
    parser.add_argument('--target_fmt', default='%f', help="Format of data in target file (e.g. '%d', '%f'). Defaults to float '%f'")
    args = parser.parse_args()
    main(args)

