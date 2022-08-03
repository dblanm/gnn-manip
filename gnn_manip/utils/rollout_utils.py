# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#
import os
import torch
import argparse
import numpy as np

from gnn_manip.models import EncProcDecGNN
from gnn_manip.utils.coffee_dataset import CoffeeTestDataset
from gnn_manip.utils.plot_utils import plot_multiple_nodes


def compute_rollout(dataset, model, args):
    if args.cma_traj is not None:
        nof_steps = dataset.time_steps
    else:
        nof_steps = dataset.time_steps - args.k_steps
    nof_particles = dataset[0][0].shape[1]
    rigid_particles = dataset[0][0][0, :, dataset.material_id] == 1
    coffee_particles = dataset[0][0][0, :, dataset.material_id] == 0
    data_dim = dataset[0][0].shape[2]

    if args.cma_traj is not None:
        npy_trajectory = np.load(args.cma_traj)
        ty_init = [0.5, 0.5, 0.4]
        # Define the initial rigid body particles position
        rigid_particles_attr = dataset[0][0][-1, rigid_particles, :]
        rigid_particles_pos = rigid_particles_attr[:, dataset.cartesian_idx]
        # Get the trajectory
        trajectory = get_rigid_body_trajectory_from_diff(npy_trajectory, nof_steps, ty_init, rigid_particles_pos)
    else:
        groundtruth = extract_groundtruth(dataset, nof_steps, nof_particles, data_dim)

    # Use model to predict the dynamics
    prediction = np.zeros((nof_steps, nof_particles, data_dim))
    current_state = dataset[0]
    for i in range(nof_steps):
        with torch.no_grad():
            new_rigid_state = current_state[0][-1, rigid_particles, :].clone()
            if args.cma_traj is not None:
                new_rigid_state[:, dataset.control_idx] = trajectory[i] - current_state[0][-1, rigid_particles, :][:,
                                                                          dataset.cartesian_idx]
            else:
                new_rigid_state[:, dataset.control_idx] = groundtruth[i, rigid_particles, :][:, dataset.control_idx]
            # Replace rigid body state with trajectory
            current_state[0][-1, rigid_particles, :] = new_rigid_state
            prediction[i, :, :] = np.array(current_state[0][-1].numpy())

            # Predict the next position and update the state accordingly
            next_pos = perform_prediction(dataset, model, current_state, args.device)

            current_state[0][:-1, :, :] = current_state[0][1:, :, :].clone()
            current_state[0][-1, :, dataset.cartesian_idx] = next_pos.detach().clone()

            # Replace predicted rigid-body with control trajectory
            if args.cma_traj is not None:
                new_rigid_state[:, dataset.cartesian_idx] = trajectory[i]
            else:
                new_rigid_state[:, dataset.cartesian_idx] = groundtruth[i, rigid_particles, :][:, dataset.cartesian_idx]
            current_state[0][-1, rigid_particles, :] = new_rigid_state.clone()

            if args.plot:
                plot_rollout_timestep(current_state, coffee_particles, rigid_particles,
                                      dataset.cartesian_idx, args.output, i)

    return prediction


def plot_rollout_timestep(current_state, coffee_idx, rigid_idx, cartesian_idx, plot_dir, it):
    coffee_particles = current_state[0][-1, coffee_idx, :]
    coffee_pos = coffee_particles[:, cartesian_idx]
    rigid_body_particles = current_state[0][-1, rigid_idx, :]
    rigid_pos = rigid_body_particles[:, cartesian_idx]
    str_it = str(it)
    if it < 10:
        str_it = "00" + str(it)
    elif it < 100:
        str_it = "0" + str(it)
    plot_multiple_nodes(coffee_pos, rigid_pos, title=plot_dir + '/CMA_' + str_it + "_")


def extract_groundtruth(dataset, nof_steps, nof_particles, data_dim):
    groundtruth = np.zeros((nof_steps, nof_particles, data_dim))

    for i in range(nof_steps):
        state, _ = dataset[i]  # Get observation and next position
        groundtruth[i, :, :] = np.array(state[-1])

    groundtruth = torch.from_numpy(groundtruth).float()

    return groundtruth


def perform_prediction(dataset, model, current_state, device):
    batch = dataset.graph_attr.process_collate([current_state])
    x = batch[0].to(device)
    edge_attr = batch[1].to(device)
    edge_index = batch[2].to(device)

    # Predict acceleration of particles
    pred_acc = model.forward(x, edge_attr, edge_index).to('cpu')

    # Get next positions and update the state accordingly
    next_pos = get_position_from_prediction(dataset.stats, dataset.cartesian_idx,
                                            pred_acc, current_state[0])
    return next_pos


def get_dataset(args, folder_new=None, sim_id=None):
    if folder_new == None:
        print("Using dataset:=", args.dir)
        test_dataset = CoffeeTestDataset(directory=args.dir, split='test', k=6, conn_r=0.015,
                                         noise=None, use_control=args.use_control, sim_id=args.sim_id)
    else:
        print("Using dataset:=", folder_new, " test ID=", sim_id)
        test_dataset = CoffeeTestDataset(directory=folder_new, split='test', k=6, conn_r=0.015,
                                         noise=None, use_control=args.use_control, sim_id=sim_id)
    print("Data loaded...")
    return test_dataset


def get_model(dataset, args):
    node_dim, edge_dim, out_dim = None, None, None
    for obs, tgt in dataset:
        nodes, edge_attr, _, _, acc = dataset.graph_attr.process(obs, tgt)
        node_dim = len(nodes[0])
        edge_dim = len(edge_attr[0])
        out_dim  = len(acc[0])
        break
    # Initialize model
    #node_dim, edge_dim, out_dim = (15, 3, 2) if not args.use_control else (17, 3, 2)
    gnn_model = EncProcDecGNN(node_dim=node_dim, edge_dim=edge_dim, out_dim=out_dim,
                              hidden_size=128, num_layers=2, m_steps=args.m_steps).to(args.device)
    print("Model created")
    # First load to cPU
    state_dict = torch.load(args.model, map_location=args.device)
    # Load model
    gnn_model.load_state_dict(state_dict)
    print('Loaded model...')
    return gnn_model


# Modified from get_position_from_prediction from cloth_dataset
def get_position_from_prediction(stats, cartesian_idx, pred_acc, obs_seq):
    # pred_acc:   N x pos_dim
    # obs_seq:    k x N x data_dim
    # Unnormalize acceleration prediction
    acc_unnorm = (pred_acc * stats['acceleration_std'].to(pred_acc.device))\
                    + stats['acceleration_mean'].to(pred_acc.device)
    # Get current position and previous velocity
    last_pos = obs_seq[-1, :, cartesian_idx]
    last_vel = last_pos - obs_seq[-2, :, cartesian_idx]
    # Update velocity to get current velocity
    vel_pred = last_vel + acc_unnorm
    # Update position to get next position
    pos_pred = last_pos + vel_pred
    return pos_pred


def get_rigid_body_trajectory_from_diff(trajectory, horizon, ty_init, rigid_particles):
    """ Take as input the CMA-ES trajectory,
    the CMA-ES values are the difference in the rotation and  translation from the previous rot/translation.
    """
    traj_rot = trajectory[:, 0]
    traj_ty = trajectory[:, 1]

    rigid_body_traj = []
    for i in range(horizon):
        rigid_body_traj.append(compute_particles_tmatrix(traj_rot[i], traj_ty[i], ty_init, rigid_particles))

    rigid_body_tensor = torch.stack(rigid_body_traj)

    # acc_loss = self.compute_acc_loss(actions)
    return rigid_body_tensor


def compute_particles_tmatrix( rotation, translation, ty_init, rigid_particles):
    """Compute the particles transformation matrix from their initial state
    rotation around the X axis? in radians, translation vector [x y z]"""
    # Create the matrices
    w_R_0 = torch.tensor([[1, 0, 0], [0, np.cos(rotation), -np.sin(rotation)],
                          [0, np.sin(rotation), np.cos(rotation)], [0, 0, 0]], dtype=torch.float32)

    w_p_0 = torch.tensor([[ty_init[0]], [ty_init[1] + translation],
                          [ty_init[2]], [1]], dtype=torch.float32)

    w_T_0 = torch.hstack((w_R_0, w_p_0))  # Create the transformation matrix, shape 4x4

    # Define the translation from the rigid body frame to the particles
    # Also need to change the y-coordinate (1) to the z-coordinate (2)
    init_state = torch.ones((4, rigid_particles.shape[0]))  # Shape 4x828
    init_state[0, :] = ty_init[0] - rigid_particles[:, 0]
    init_state[1, :] = ty_init[1] - rigid_particles[:, 2]
    init_state[2, :] = ty_init[2] - rigid_particles[:, 1]

    # Apply the transformation matrix
    transformed_particles = w_T_0 @ init_state  # 4x4 @ 4x828, should give 4x828
    # Change back the particles to X Z Y
    end_particles = torch.zeros((rigid_particles.shape[0], rigid_particles.shape[1]))
    end_particles[:, 0] = transformed_particles[0, :]
    end_particles[:, 2] = transformed_particles[1, :]
    end_particles[:, 1] = transformed_particles[2, :]

    return end_particles


def create_arg_parser():
    # Get the current directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    blender_file_path = dir_path + "../../examples/render_dyn_blender.py"

    parser = argparse.ArgumentParser(description='Granular manipulation argument parser.')

    # Plot arguments
    parser.add_argument('--plot', action='store_true', help='Whether to plot the rollout or not')
    parser.add_argument('--plot_dir', default=dir_path+"../../")

    # Blender arguments
    parser.add_argument('--blender_path', default='blender', help='Path to blender.exe or blender')
    parser.add_argument('--blender_file', default=blender_file_path,
                        help='Python script run by Blender in order to render the scene')
    parser.add_argument('--output', help='Path to output directory')
    parser.add_argument('--step', default=3, type=int, help='Frame step')
    parser.add_argument('--use_transparent_background', action='store_true', help="Render transparent background")
    parser.add_argument('--hide_rigids', action='store_true', help='Hide rigid particles')
    parser.add_argument('--hide_background_objects', action='store_true',
                        help='Hide background objects that is the container and table')
    parser.add_argument('--coffee_color', default='0xcc9200', help='Color of coffee')
    parser.add_argument('--diameter', default=0.002, type=float,
                        help='Diameter of particles (d=0.002 -> normal, d=0.004 -> large)')
    parser.add_argument('--res', default=512, type=int, help="Resolution of output images (res, res)")
    parser.add_argument('--camera_idx', default=0, type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help="Camera position/orientation to be used. 0 = Corner view, 1 = Front view whole, 2 = Front view container, 3 = Top view container, 4 = Back view container, 5 = Top view container rotated, 6 =  Back view angled, 7 = Back view wide")
    parser.add_argument('--camera_position', default=None, nargs='*', type=float,
                        help="Custom camera position and orientation (x_posititon, y_position, z_position, x_rotation, y_rotation, z_rotation)")
    parser.add_argument('--save_ffmpeg', action='store_true',
                        help='Save as FFmpeg video instead of individual PNG frames.')

    # Architecture/model related arguments
    parser.add_argument('-c', '--use_control', action='store_true', help='use control inputs')
    parser.add_argument('-pr', '--predict_rigids', action='store_true',
                        help='use predicted positions for rigid body particles')
    parser.add_argument('--k_steps', type=int, default=6,
                        help='Previous k positions to be used to compute node attributes')
    parser.add_argument('--conn_r', type=float, default=0.015,
                        help='Connectivity radius for graph formation of particles')
    parser.add_argument('--max_neighbours', type=int, default=20,
                        help='Maximum number of neighbors for each node in graph')
    parser.add_argument('--m_steps', type=int, default=10, help='Number of message passing steps (GN blocks)')
    parser.add_argument('--hidden_size', type=int, default=128, help='Size of the hidden layer in MLPs')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in MLPs')

    # Data related arguments
    parser.add_argument('-d', '--dir', help='dataset directory')
    parser.add_argument('-m', '--model', help='path to model file to be used for rollout generation')
    parser.add_argument('-rd', '--rollout_dir', default='',
                        help='directory to which the rollout results are to be saved')
    parser.add_argument('--sim_id', default=1, type=int,
                        help="index of simulation in test dataset for which you want the rollout for, the data is named as 'particles_{06d:sim_id}.csv'")
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda:0', 'cuda:1'],
                        help="Training device, the options are: 'cpu', 'cuda:0', 'cuda:1'")

    # CMA-ES Parameters
    parser.add_argument('--cma_alpha', default=0.0000, type=float, help='regularisation term to apply in the cost function')
    parser.add_argument('--cma_beta', default=1000.0, type=float, help='regularisation term for the W2 distance')
    parser.add_argument('--cma_gamma', default=0.05, type=float, help='regularisation term to apply in the cost function')
    parser.add_argument('--cma_penalty', default=0.0, type=float, help='penalty term for the loss for pouring ouside the boundaries')
    parser.add_argument('--cma_rho', default=0.0, type=float, help='regularisation term for the cup end position')
    parser.add_argument('--cma_iter', type=int, default=10, help='Saved CMA-ES trajectory npy file.')
    parser.add_argument('--cma_rand', type=int, default=1234, help='CMA-ES random number.')
    parser.add_argument('--cma_var', type=float, default=1.5, help='Variance to use for CMA-ES.')
    parser.add_argument('--cma_popsize', default=10, type=int, help='Population size of CMA-ES')
    parser.add_argument('--cma_traj', default=None, help='Saved CMA-ES trajectory npy file.')

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    return args

