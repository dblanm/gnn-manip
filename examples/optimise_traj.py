# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#
import os
import time
from datetime import datetime
import argparse
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams

import torch
import numpy as np

from gnn_manip.utils.coffee_dataset import CoffeeTestDataset
from gnn_manip.models import EncProcDecGNN
from gnn_manip.utils.traj_utils import TrajectoryCMAsolver, InterpolatedCMAsolver, save_loss_results
from gnn_manip.utils.plot_utils import plot_trajectory
from gnn_manip.utils.rollout_utils import get_position_from_prediction, get_dataset, get_model

matplotlib.use('Agg')
rcParams['text.usetex'] = False
rcParams['figure.subplot.left'] = 0.15
rcParams['figure.subplot.right'] = 0.95
rcParams['figure.subplot.bottom'] = 0.12
rcParams['figure.subplot.top'] = 0.95
# rcParams['figure.figsize'] = 6, 6
rcParams['figure.dpi'] = 100
# plt.ion()


def plot_multiple_nodes(sand_particles, rigid_particles, title, desired=None):
    # Plot YZ
    fig, ax4 = plt.subplots()
    ax4.scatter(sand_particles[:, 2], sand_particles[:, 1], color='darkorange', zorder=4, label='Sand Particles')
    ax4.scatter(rigid_particles[:, 2], rigid_particles[:, 1], color='grey', zorder=3, label='Rigid Particles')
    if desired is not None:
        ax4.scatter(desired[:, 2], desired[:, 1], color='darkkhaki', zorder=2, label='Desired')
    ax4.set_xlabel("Y")
    ax4.set_ylabel("Z")
    ax4.set_xlim(0.0, 1.0)
    ax4.set_ylim(0.0, 0.8)
    ax4.legend(frameon=True)
    ax4.axhline(y=0.1,xmin=0.4, xmax=0.6)
    ax4.axvline(x=0.4, ymin=0.1, ymax=0.3)
    ax4.axvline(x=0.6, ymin=0.1, ymax=0.3)
    plt.savefig(title+'particles_YZ.png', dpi=100)
    plt.close(fig)


def test_cma_trajectory(args, dir_origin, t_optim, optimal_traj, sim_id=1):

    # Create the folder for the plot
    if args.cma_traj  is not None:
        plot_dir = args.cma_traj + '/sim_'+str(sim_id)+'_cma_rollout/'
    else:
        plot_dir = dir_origin + '/sim_'+str(sim_id)+'_cma_rollout/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Flags for test time
    sing_time = []

    current_state = (t_optim.initial_state[0].clone(), t_optim.initial_state[1].clone())

    trajectory, actions = t_optim.get_rigid_body_trajectory_from_diff(optimal_traj, demo=True)

    # Compute the initial cost
    sand_particles = current_state[0][-1, t_optim.coffee_particles_idx, :]
    # sand_pos = sand_particles[:, t_optim.graph_attr.cartesian_idx]
    initial_sand_pos = sand_particles[:, t_optim.graph_attr.cartesian_idx]

    _, initial_wass_loss, _, _, _, _ = t_optim.compute_loss(initial_sand_pos,
                                                            np.zeros((args.total_steps-200, 3)))
    print("Initial Wasserstein loss (initial position VS goal position)=", initial_wass_loss)

    # sand_trajectory = []
    cup_states = []
    sand_states = []
    start = time.time()
    for i in range(t_optim.horizon):
        with torch.no_grad():
            # Get the control point from the optimized trajectory
            new_rigid_state = current_state[0][-1, t_optim.rigid_particles_idx, :].clone()
            # Control attributes are  =  next_pos - current_pos
            if i >= trajectory.shape[0]:  # In this case we don't modify the next state
                new_rigid_state[:, t_optim.graph_attr.control_idx] = current_state[0][-1, t_optim.rigid_particles_idx, :][:,t_optim.graph_attr.cartesian_idx]
            else:
                new_rigid_state[:, t_optim.graph_attr.control_idx] = trajectory[i] - current_state[0][-1, t_optim.rigid_particles_idx, :][:,t_optim.graph_attr.cartesian_idx]
                cup_states.append(new_rigid_state[:, t_optim.graph_attr.cartesian_idx])
                sand_state = current_state[0][-1, t_optim.coffee_particles_idx, :]
                sand_states.append(sand_state[:, t_optim.graph_attr.cartesian_idx])
            current_state[0][-1, t_optim.rigid_particles_idx, :] = new_rigid_state.clone()

            batch = t_optim.graph_attr.process_collate([current_state])
            x          = batch[0].to(args.device)
            edge_attr  = batch[1].to(args.device)
            edge_index = batch[2].to(args.device)
            start_pred_time = time.time()
            # Predict acceleration of particles
            pred_acc   = t_optim.model.forward(x, edge_attr, edge_index).to('cpu').detach()
            sing_pred_time = time.time() - start_pred_time
            sing_time.append(sing_pred_time)
            # Get next positions and update the state accordingly
            next_pos   = get_position_from_prediction(t_optim.graph_attr.stats,
                                                      t_optim.graph_attr.cartesian_idx, pred_acc, current_state[0])
            current_state[0][:-1,:,:] = current_state[0][1:,:,:].clone()
            current_state[0][-1,:,t_optim.graph_attr.cartesian_idx] = next_pos.clone()

            # Replace predicted rigid-body positions with control trajectory
            if i < trajectory.shape[0]:
                new_rigid_state[:, t_optim.graph_attr.cartesian_idx] = trajectory[i]
            current_state[0][-1, t_optim.rigid_particles_idx, :] = new_rigid_state.clone()

            # Get the rigid body particles and sand particles and plot them
            sand_particles = current_state[0][-1, t_optim.coffee_particles_idx, :]
            sand_pos = sand_particles[:, t_optim.graph_attr.cartesian_idx]
            rigid_body_particles = current_state[0][-1, t_optim.rigid_particles_idx, :]
            rigid_pos = rigid_body_particles[:, t_optim.graph_attr.cartesian_idx]
            str_it = str(i)
            if i < 10:
                str_it = "00" + str(i)
            elif i < 100:
                str_it = "0" + str(i)
            plot_multiple_nodes(sand_pos, rigid_pos, title=plot_dir+'/CMA_'+str_it+"_", desired=t_optim.desired_pos)

    time_all = time.time() - start
    single_time = np.array(sing_time).mean()
    # Measure the loss between last state VS desired particles position
    sand_particles = current_state[0][-1, t_optim.coffee_particles_idx, :]
    sand_pos = sand_particles[:, t_optim.graph_attr.cartesian_idx]

    loss, wasserstein_loss, vel_loss, \
    acc_loss, bound_loss, theta_loss = t_optim.compute_loss(sand_pos, actions, cup_states, sand_states)

    save_loss_results(loss, initial_wass_loss, wasserstein_loss, vel_loss,
                      acc_loss, bound_loss, theta_loss, dir_origin, sim_id,
                      time_all=time_all, time_single=single_time)


def setup_files(args, plot_dir=None):

    # Set the plot directory
    dateTimeObj = datetime.now()
    main_folder = "runs/"
    date_test = str(dateTimeObj.month) + "m_" + str(dateTimeObj.day) + "d_" + str(dateTimeObj.hour) \
                + "h_" + str(dateTimeObj.minute) + "m_" + str(dateTimeObj.second) + "s"
    if plot_dir is None:
        plot_dir = main_folder + date_test
    else:
        plot_dir = plot_dir
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Create a text file with the parameters of the optimisation
    with open(plot_dir+'/params.txt', 'w') as fd:
        text = "scale_rot:=[" + str(args.scale_rot) + "]\n scale_ty:=[" + str(args.scale_ty) + "]\n" + \
               "max_rot:=[" + str(args.max_rot) + "]\n max_ty:=[" + str(args.max_ty) + "]\n" + \
               "cma_alpha:=[" + str(args.cma_alpha) + "]\n cma_beta:=[" + str(args.cma_beta) + "]\n" + \
               "cma_gamma:=[" + str(args.cma_gamma) + "]\n cma_penalty:=[" + str(args.cma_penalty) + "]\n" + \
               "cma_rho:=[" + str(args.cma_rho) + "]\n cma_var:=[" + str(args.cma_var) + "]\n" + \
               "cma_popsize:=[" + str(args.cma_popsize) + "]\n cma_rand:=[" + str(args.cma_rand) + "]\n" + \
               "traj points:=[" + str(args.traj_points) + "]\n" + \
               "cma_iter:=[" + str(args.cma_iter) + "]\n\n total_steps:=[" + str(args.total_steps) + "]\n\n"
        fd.write(text)

    return plot_dir


def run_trajectory_optim(args, opt, plot_dir):

    for i in args.test_list:
        print("Running test case ID=", i)
        test_folder = "/home/mulerod1/projects/sand_simulation/dataset/coffee-new3D-v2/"
        #test_folder = "/scratch/work/mulerod1/gnn-manip/sand_simulation/dataset/coffee-new3D-v2/"

        # Get the goal position
        goal_dataset = get_dataset(args, folder_new=test_folder, sim_id=i)
        sand_particles_idx = goal_dataset[len(goal_dataset)-1][0][-1,:,goal_dataset.graph_attr.material_idx[0]] == 0
        sand_data = goal_dataset[len(goal_dataset)-1][0][0, sand_particles_idx, :]
        desired_pos = sand_data[:, goal_dataset.cartesian_idx]

        if args.cma_traj is not None:
            optimal_traj = np.load(args.cma_traj+'/trajectory_feasible_sim_'+str(i)+'.npy')

            opt.desired_pos = desired_pos
            # Test the trajectory and plot it
            test_cma_trajectory(args, plot_dir, opt, optimal_traj, sim_id=i)
        else:
            # Find the optimal trajectory
            xopt, es = opt.optimize_trajectory(desired_pos)

            # Save, plot and test the optimal trajectory
            save_plot_and_test_trajectory(args, xopt, opt, plot_dir, i,
                                          savename=plot_dir+'/trajectory_sim_',
                                          title='Original vs Best solution',
                                          savetitle='/best_solution_sim_'+str(i))

            # Same for the best feasible solution if we found one
            if es.best_feasible.info is not None:
                best_feasible = np.array(es.best_feasible.info['x'])
                print("Best feasible constraints=", es.best_feasible.info['g'])
                print("Best feasible value=", es.best_feasible.f)

                # Save, plot and test the best feasible trajectory
                save_plot_and_test_trajectory(args, best_feasible, opt, plot_dir+'/feasible/', i,
                                              savename=plot_dir+'/trajectory_feasible_sim_',
                                              title='Original vs Best feasible solution',
                                              savetitle='/best_feasible_solution_sim_' + str(i))


def save_plot_and_test_trajectory(args, traj, opt, plot_dir, test_id, savename, title, savetitle):
    # Save the trajectory
    traj_rot, traj_ty = opt.interpolate_trajectory(traj)
    optimal_traj = np.zeros((len(traj_rot), 2))
    optimal_traj[:, 0] = traj_rot
    optimal_traj[:, 1] = traj_ty

    np.save(savename + str(test_id) + '.npy', optimal_traj)

    # Plot the trajectory
    plot_vs_initial_traj(opt, traj, plot_dir, title=title, savetitle=savetitle)

    # Test the trajectory
    test_cma_trajectory(args, plot_dir, opt, optimal_traj, sim_id=test_id)


def plot_vs_initial_traj(opt, traj, plot_dir, title, savetitle):
    traj_rot, traj_ty = opt.interpolate_trajectory(traj)
    optimal_traj = np.zeros((len(traj_rot), 2))
    optimal_traj[:, 0] = traj_rot
    optimal_traj[:, 1] = traj_ty

    # Plot the trajectory
    initial_traj = np.zeros(int(opt.sample_traj.shape[0] * 2))
    initial_traj[:opt.sample_traj.shape[0]] = opt.sample_traj[:, 0]
    initial_traj[opt.sample_traj.shape[0]:] = opt.sample_traj[:, 1]
    init_rot, init_ty = opt.interpolate_trajectory(initial_traj)
    init_rot = np.rad2deg(init_rot)

    # Test that the trajectory interpolation gives similar values to the sample trajectory
    traj_rot_deg = np.rad2deg(traj_rot)
    plot_trajectory(init_rot, init_ty, traj_rot_deg, traj_ty, title=title, save_name=plot_dir + savetitle)


def main_optimal_trajectory(args):
    start_idx = 0
    angle_max = args.angle_constraint  # max/min radians of robot

    rigid_body_ref = [0.5, 0.5, 0.4]
    rotation_init = 180

    test_dataset = get_dataset(args)  # Get dataset

    gnn_model = get_model(test_dataset, args)  # Get and load the model

    if args.interpolate:
        print("Running InterpolatedCMASolver")
        opt = InterpolatedCMAsolver(gnn_model, test_dataset.graph_attr, test_dataset[start_idx],
                                    rotation_init, rigid_body_ref, scale_rot=args.scale_rot, scale_ty=args.scale_ty,
                                    max_rot=args.max_rot, max_ty=args.max_ty,
                                    alpha=args.cma_alpha, beta=args.cma_beta, gamma=args.cma_gamma,
                                    penalty=args.cma_penalty,
                                    rho=args.cma_rho, device=args.device, total_steps=args.total_steps,
                                    cma_iter=args.cma_iter, cma_var=args.cma_var, cma_rand=args.cma_rand,
                                    cma_popsize=args.cma_popsize, traj_points=args.traj_points)
    else:
        print("Running TrajectoryCMASolver")
        opt = TrajectoryCMAsolver(gnn_model, test_dataset.graph_attr, test_dataset[start_idx],
                                  rotation_init, rigid_body_ref, scale_rot=args.scale_rot, scale_ty=args.scale_ty,
                                  max_rot=args.max_rot, max_ty=args.max_ty,
                                  alpha=args.cma_alpha, beta=args.cma_beta, gamma=args.cma_gamma,
                                  penalty=args.cma_penalty,
                                  rho=args.cma_rho, device=args.device, total_steps=args.total_steps,
                                  cma_iter=args.cma_iter, cma_var=args.cma_var, cma_rand=args.cma_rand,
                                  cma_popsize=args.cma_popsize)

    if args.sample_traj is not None:
        initial_trajectory = np.load(args.sample_traj)
        opt.set_sample_traj(initial_trajectory)

    else:
        sample_trajectory = np.zeros((test_dataset.time_steps+1, 2))  # The initial trajectories are of 301
        opt.sample_traj = sample_trajectory

    # Set-up the folder to save the trajectories, plots and write the training/test arguments
    plot_dir = setup_files(args)
    setup_files(args, plot_dir=plot_dir+"/feasible")

    # initial_traj = np.zeros(int(opt.sample_traj.shape[0] * 2))
    # initial_traj[:opt.sample_traj.shape[0]] = opt.sample_traj[:, 0]
    # initial_traj[opt.sample_traj.shape[0]:] = opt.sample_traj[:, 1]
    #
    # plot_vs_initial_traj(opt, initial_traj, plot_dir, title='Original vs Original pchip solution',
    #                      savetitle='/test_interp')

    run_trajectory_optim(args, opt, plot_dir)


def get_arguments():
    parser = argparse.ArgumentParser(description='Generates rollout prediction for given model.')
    parser.add_argument('-c', '--use_control', action='store_true', help='use control inputs')
    parser.add_argument('-d', '--dir', help='dataset directory')
    parser.add_argument('--desired', help='CSV file with the desired position of the sand.')
    parser.add_argument('-m', '--model', help='path to model file to be used for rollout generation')
    parser.add_argument('--sim_id', default=1, type=int, help="index of simulation in test dataset for which you want the rollout for, the data is named as 'particles_{06d:sim_id}.csv'")
    parser.add_argument('--device', default='cuda:0', choices=['cpu', 'cuda:0', 'cuda:1'], help="Training device, the options are: 'cpu', 'cuda:0', 'cuda:1'")

    # Trajectory parameters
    parser.add_argument('--sample_traj', default='/home/mulerod1/projects/sand_simulation/examples/sine_traj.npy',
                        help='file to load as sample trajectory for initialise CMA-ES')
    parser.add_argument('--scale_rot', default=np.pi, type=float, help='scale applied to the CMA-ES rotation trajectory')
    parser.add_argument('--scale_ty', default=0.110335, type=float, help='scale applied to the CMA-ES translation trajectory')
    parser.add_argument('--max_rot', default=2.1973, type=float, help='scale applied to the CMA-ES rotation trajectory')
    parser.add_argument('--max_ty', default=0.002, type=float, help='scale applied to the CMA-ES translation trajectory')
    parser.add_argument('--angle_constraint', default=2.8973, type=float, help='Max/Min angle constraint for manipulator')
    parser.add_argument('--traj_points', default=50, type=int, help='Scaling factor for computing number of trajectory points.')
    parser.add_argument('--interpolate', action='store_true', help='Whether to use the interpolated version or delta trajectory')
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

    parser.add_argument('--total_steps', default=300, type=int, help='total steps in the rollout')
    parser.add_argument('--test_list', nargs="+", type=int, default=[1, 2, 3, 4], help='test cases')
    parser.add_argument('--m_steps', type=int, default=10, help='Specify the message passing steps.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    """ Usage: python rollout_sand_dyn.py
    -c -d /home/mulerod1/projects/sand_manip/dataset/still-cup/
    -m /home/mulerod1/projects/sand_manip/models/model3D_v1_002000.pth 
    -rd rollout_dir --sim_id 1 --desired /home/mulerod1/projects/sand_manip/dataset/still-cup/sand_at_rest.csv
    --scale_rot 20 --scale_ty 0.1"""

    print(torch.cuda.device_count())
    args = get_arguments()
    print('Using device:', args.device)
    main_optimal_trajectory(args)
