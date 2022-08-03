# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#

import cma
import torch
import numpy as np
import geomloss
from scipy.interpolate import interp1d, pchip_interpolate

from gnn_manip.utils.rollout_utils import get_position_from_prediction


class CMAESolver:
    def __init__(self, model, graph_attr, state_init,
                 rx_init, ty_init, scale_rot, scale_ty,
                 alpha, beta, gamma, penalty, rho, device,
                 cma_iter=10,  cma_var=0.5, cma_popsize=21, cma_rand=1234,
                 max_rot=1.9337, max_ty=6.67e-4, total_steps=300, traj_points=10):

        print("Creating Trajectory Optimisation...")
        print("Using scale_rot=", scale_rot)
        print("Using scale_ty=", scale_ty)
        print("Using alpha=", alpha)
        print("Using beta=", beta)
        print("Using gamma=", gamma)
        print("Using penalty=", penalty)
        print("Using rho=", rho)
        print("Using cma popsize=", cma_popsize)
        print("Using cma var=", cma_var)
        self.initial_state = state_init
        self.rx_init = np.deg2rad(rx_init)
        self.ty_init = ty_init
        self.sample_traj = None
        self.desired_pos = None
        self.model = model
        self.graph_attr = graph_attr
        self.horizon = total_steps  # Define the horizon for which we will run the traj
        self.device = device

        # Define the scaling factor for the CMA-ES rotation and translation
        self.scale_rot = scale_rot
        self.scale_ty = scale_ty
        # Number of points the trajectory is divided into
        self.nr_traj_points = traj_points
        self.traj_points = int(self.horizon/self.nr_traj_points)

        # Define the parameters for the loss
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.penalty = penalty
        self.rho = rho
        self.max_rot = np.deg2rad(max_rot)  # Maximum rotation
        self.max_ty = max_ty  # Maximum translation
        self.total_steps = total_steps

        # Limits for the boundaries penalties
        self.left_limit = 0.3
        self.right_limit = 0.7
        self.scale_ty = (self.ty_init[0] - self.left_limit)/self.scale_rot
        self.rotation_limit = 2.8973

        # Define the initial position of the rigid-body particles
        self.rigid_particles_idx = self.initial_state[0][-1,:,self.graph_attr.material_idx[0]] == 1
        self.coffee_particles_idx = self.initial_state[0][-1, :, self.graph_attr.material_idx[0]] == 0
        rigid_particles = self.initial_state[0][-1, self.rigid_particles_idx, :]
        self.rigid_particles = rigid_particles[:, self.graph_attr.cartesian_idx]
        self.loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=.05)  # Define the loss function

        # Define the CMA-ES optimizer options
        self.cma_options = cma.evolution_strategy.CMAOptions()
        self.cma_options['seed'] = cma_rand
        self.cma_options['maxiter'] = cma_iter  # 50 should suffice
        self.cma_options['popsize'] = cma_popsize
        self.cma_initial_var = cma_var

    def optimize_trajectory(self, desired_position):
        raise NotImplementedError()

    def interpolate_trajectory(self, x):
        raise NotImplementedError()

    def set_sample_traj(self, sample_traj):
        raise NotImplementedError()

    def get_rigid_body_trajectory_from_diff(self, x, demo=False):
        """ Take as input the CMA-ES trajectory,
        the CMA-ES values are the difference in the rotation and  translation from the previous rot/translation.
        """
        if demo:
            traj_rot = x[:, 0]
            traj_ty = x[:, 1]
        else:
            traj_rot, traj_ty = self.interpolate_trajectory(x)

        rigid_body_traj = [self.compute_particles_tmatrix(traj_rot[i], traj_ty[i]) for i in range(self.horizon)]

        rigid_body_tensor = torch.stack(rigid_body_traj)
        actions = np.zeros((self.horizon, 2))
        actions[:, 0] = traj_rot
        actions[:, 1] = traj_ty
        return rigid_body_tensor, actions

    def compute_loss(self, end_position, actions, cup_states=None, coffee_states=None, x=None):
        raise NotImplementedError()

    def compute_acc_loss(self, acc):
        raise NotImplementedError()

    def compute_vel_loss(self, vel):
        raise NotImplementedError()

    def cma_objective(self, x):
        """
        Function that runs the model for the horizon using as control the optimized trajectory points
        x and computes the loss between the final state and desired position.
        """
        current_state = (self.initial_state[0].clone(), self.initial_state[1].clone())
        trajectory, actions = self.get_rigid_body_trajectory_from_diff(x)
        cup_states = []
        coffee_states = []
        for i in range(self.horizon):
            with torch.no_grad():
                # Get the control point from the optimized trajectory
                new_rigid_state = current_state[0][-1, self.rigid_particles_idx, :].clone()
                # Control attributes are  =  next_pos - current_pos
                if i >= trajectory.shape[0]:  # In this case we don't modify the next state
                    new_rigid_state[:, self.graph_attr.control_idx] = current_state[0][-1, self.rigid_particles_idx, :][:,self.graph_attr.cartesian_idx]
                else:
                    new_rigid_state[:, self.graph_attr.control_idx] = trajectory[i] - current_state[0][-1, self.rigid_particles_idx, :][:,self.graph_attr.cartesian_idx]
                    cup_states.append(new_rigid_state[:, self.graph_attr.cartesian_idx])
                    coffee_state = current_state[0][-1, self.coffee_particles_idx, :]
                    coffee_states.append(coffee_state[:, self.graph_attr.cartesian_idx])
                current_state[0][-1, self.rigid_particles_idx, :] = new_rigid_state.clone()

                batch = self.graph_attr.process_collate([current_state])
                node_attr  = batch[0].to(self.device)
                edge_attr  = batch[1].to(self.device)
                edge_index = batch[2].to(self.device)
                # Predict acceleration of particles
                pred_acc   = self.model.forward(node_attr, edge_attr, edge_index).to('cpu').detach()
                # Get next positions and update the state accordingly
                next_pos   = get_position_from_prediction(self.graph_attr.stats,
                                                          self.graph_attr.cartesian_idx, pred_acc, current_state[0])
                current_state[0][:-1, :, :] = current_state[0][1:, :, :].clone()
                current_state[0][-1, :, self.graph_attr.cartesian_idx] = next_pos.clone()

                # Replace predicted rigid-body positions with control trajectory
                if i < trajectory.shape[0]:
                    new_rigid_state[:, self.graph_attr.cartesian_idx] = trajectory[i]
                current_state[0][-1, self.rigid_particles_idx, :] = new_rigid_state.clone()

        # Measure the loss between last state VS desired particles position
        coffee_particles = current_state[0][-1, self.coffee_particles_idx, :]
        end_position = coffee_particles[:, self.graph_attr.cartesian_idx]

        loss, _, _, _, _, _ = self.compute_loss(end_position, actions, cup_states, coffee_states, x)
        return loss

    def compute_vel_acc(self, actions):
        vel = actions[1:, :]-actions[:-1, :]
        acc = actions[2:, :] - 2*actions[1:-1, :] + actions[:-2, :]

        return vel, acc

    def compute_particles_tmatrix(self, rotation, translation):
        """Compute the particles transformation matrix from their initial state
        rotation around the X axis? in radians, translation vector [x y z]"""
        # Create the matrices
        w_R_0 = torch.tensor([[1, 0, 0], [0, np.cos(rotation), -np.sin(rotation)],
                              [0, np.sin(rotation), np.cos(rotation)], [0, 0, 0]], dtype=torch.float32)

        w_p_0 = torch.tensor([[self.ty_init[0]], [self.ty_init[1] + translation],
                              [self.ty_init[2]], [1]], dtype=torch.float32)

        w_T_0 = torch.hstack((w_R_0, w_p_0))  # Create the transformation matrix, shape 4x4

        # Define the translation from the rigid body frame to the particles
        # Also need to change the y-coordinate (1) to the z-coordinate (2)
        init_state = torch.ones((4, self.rigid_particles.shape[0]))  # Shape 4x828
        init_state[0, :] = self.ty_init[0] - self.rigid_particles[:, 0]
        init_state[1, :] = self.ty_init[1] - self.rigid_particles[:, 2]
        init_state[2, :] = self.ty_init[2] - self.rigid_particles[:, 1]

        # Apply the transformation matrix
        transformed_particles = w_T_0 @ init_state  # 4x4 @ 4x828, should give 4x828
        # Change back the particles to X Z Y
        end_particles = torch.zeros((self.rigid_particles.shape[0], self.rigid_particles.shape[1]))
        end_particles[:, 0] = transformed_particles[0, :]
        end_particles[:, 2] = transformed_particles[1, :]
        end_particles[:, 1] = transformed_particles[2, :]

        return end_particles


class TrajectoryCMAsolver(CMAESolver):

    def set_sample_traj(self, sample_traj):
        initial_trajectory = sample_traj[2:] - sample_traj[1:-1]
        rotation_scaled = np.deg2rad(initial_trajectory[:, 0] / self.scale_rot)
        translation_scaled = initial_trajectory[:, 1] / self.scale_ty
        sample_trajectory = np.stack((rotation_scaled, translation_scaled)).T
        self.sample_traj = sample_trajectory

    def interpolate_trajectory(self, x):

        prev_rotx = self.rx_init  # Initially 180
        prev_ty = 0.0

        traj_rot = [self.rx_init]
        traj_ty = [0.0]

        for i in range(0, self.sample_traj.shape[0]):
            # Apply the scaling factors and clip the values to the maximum rotation and translation
            inc_rot = np.clip(np.deg2rad(self.scale_rot * np.rad2deg(x[i])), -self.max_rot, self.max_rot)
            inc_ty = np.clip(self.scale_ty * x[i + self.sample_traj.shape[0]], -self.max_ty, self.max_ty)
            # Update the rotation and translation
            rotx = prev_rotx + inc_rot
            ty = prev_ty + inc_ty

            traj_rot.append(rotx)
            traj_ty.append(ty)
            # Update the prev rotation and translation
            prev_rotx = rotx
            prev_ty = ty

        return traj_rot, traj_ty

    def compute_acc_loss(self, acc):
        acc_normalised = acc.copy()
        # Normalise the acceleration by the maximum rotation and translation
        acc_normalised[:, 0] = acc_normalised[:, 0] / self.max_rot
        acc_normalised[:, 1] = acc_normalised[:, 1] / self.max_ty
        # Compute the loss
        acc_loss = np.linalg.norm(acc_normalised)
        return acc_loss

    def compute_vel_loss(self, vel):
        vel_normalised = vel.copy()
        # Normalise the velocity
        vel_normalised[:, 0] = vel_normalised[:, 0] / self.max_rot
        vel_normalised[:, 1] = vel_normalised[:, 1] / self.max_ty
        vel_loss = np.linalg.norm(vel_normalised)
        return vel_loss

    def optimize_trajectory(self, desired_position):
        """ Function to optimize the trajectory, returns the optimal trajectory. """
        # Create a sample trajectory, it needs to be 2*(horizon-1), as we need rotation and translation difference
        # The CMA-ES trajectory will be t0= [180], [0]; t1= [180-dtheta], [0-dx]
        print("Starting new optimisation...")
        initial_traj = np.zeros(int(self.sample_traj.shape[0]*2))
        initial_traj[:self.sample_traj.shape[0]] = self.sample_traj[:, 0]
        initial_traj[self.sample_traj.shape[0]:] = self.sample_traj[:, 1]
        params = initial_traj.tolist()  # Shape 598 (600 -2)
        self.desired_pos = desired_position.clone()
        xopt, es = cma.fmin2(self.cma_objective, params, self.cma_initial_var, options=self.cma_options)

        return xopt, es

    def compute_boundaries_penalty(self, actions):
        # Get the rotation trajectory, which starts from 180
        rotation_traj = actions[:, 0]
        min_rot = rotation_traj.min()
        max_rot = rotation_traj.max()
        penalty = 0.0
        # The rotation  should not be less than pi- rotation_limit or more than pi+rotation_limit
        if max_rot > (self.rx_init + self.rotation_limit):
            penalty = 20.0
        elif min_rot < (self.rx_init - self.rotation_limit):
            penalty = 20.0

        return penalty

    def compute_loss(self, end_position, actions, cup_states=None, coffee_states=None, x=None):
        wasserstein_loss = self.loss(end_position, self.desired_pos).item()
        vel, acc = self.compute_vel_acc(actions)
        vel_loss = self.compute_vel_loss(vel)
        acc_loss = self.compute_acc_loss(acc)
        bound_penalty = self.compute_boundaries_penalty(actions)

        # Sum all the losses and apply the coefficients
        loss = self.beta * wasserstein_loss + self.penalty * bound_penalty + \
               self.alpha * vel_loss + self.gamma * acc_loss
        return loss, wasserstein_loss, vel_loss, acc_loss, bound_penalty, 0.0


class InterpolatedCMAsolver(CMAESolver):
    """
    For this class we should use as scale rot np.pi to ease the optimisation process.
    The bounds can only be applied in one dimension, therefore we apply the bounds on
    the rotation, setting as bound the maximum of the robot end-effector.
    Then, the scale for the translation should be  (self.ty_init-self.left_limit)/bound
    """

    def set_sample_traj(self, sample_traj):
        index_points = [i for i in range(self.nr_traj_points, sample_traj.shape[0], self.nr_traj_points)]
        # We use only # points for defining the trajectory points=horizon/traj_points=30
        traj_points = sample_traj[index_points, :]
        rotation_scaled = (np.deg2rad(traj_points[:, 0]) - self.rx_init) / self.scale_rot
        translation_scaled = (traj_points[:, 1] - self.ty_init[0]) / self.scale_ty

        self.sample_traj = np.stack((rotation_scaled, translation_scaled)).T

        self.compute_initial_acc_loss()

    def compute_initial_acc_loss(self):

        initial_traj = np.zeros(int(self.sample_traj.shape[0]*2))
        initial_traj[:self.sample_traj.shape[0]] = self.sample_traj[:, 0]  # Rotation
        initial_traj[self.sample_traj.shape[0]:] = self.sample_traj[:, 1]  # Translation
        # a = self.ineq_constraint(initial_traj)
        # Compute the trajectory
        traj_rot, traj_ty = self.interpolate_trajectory(initial_traj)
        actions = np.zeros((self.horizon, 2))
        actions[:, 0] = traj_rot
        actions[:, 1] = traj_ty
        vel, acc = self.compute_vel_acc(actions)
        vel_loss = self.compute_vel_loss(vel)
        acc_loss = self.compute_acc_loss(acc)
        print("Initial Velocity loss=", vel_loss)
        print("Initial Acceleration loss=", acc_loss)

    def optimize_trajectory(self, desired_position):
        """ Function to optimize the trajectory, returns the optimal trajectory. """
        # Create a sample trajectory, it needs to be 2*(horizon-1), as we need rotation and translation difference
        # The CMA-ES trajectory will be t0= [180], [0]; t1= [180-dtheta], [0-dx]
        print("Starting new optimisation...")
        self.cma_options['bounds'] = [-self.rotation_limit/self.scale_rot, self.rotation_limit/self.scale_rot]

        initial_traj = np.zeros(int(self.sample_traj.shape[0]*2))
        initial_traj[:self.sample_traj.shape[0]] = self.sample_traj[:, 0]
        initial_traj[self.sample_traj.shape[0]:] = self.sample_traj[:, 1]
        params = initial_traj.tolist()  # Shape 598 (600 -2)
        self.desired_pos = desired_position.clone()
        xopt, es = cma.fmin_con(self.cma_objective, params, self.cma_initial_var,
                                g=self.ineq_constraint, options=self.cma_options)
        return xopt, es
        # Optimisation with constraints
        # Equality constraints should preferably be passed as two inequality constraints like ``[h - eps, -h - eps]``,
        # with eps >= 0. When eps > 0, also feasible solution tracking can succeed.

    def compute_acc_loss(self, acc):
        acc_normalised = acc.copy()
        mean_acc = acc.mean(axis=0)
        mean_rot = 2.2e-4  # -2.2e-6
        mean_ty = 1.45e-4  # 1.4e-6
        # Normalise the acceleration by the maximum rotation and translation
        acc_normalised[:, 0] = acc_normalised[:, 0] / mean_rot
        acc_normalised[:, 1] = acc_normalised[:, 1] / mean_ty
        # Compute the loss
        acc_loss = np.linalg.norm(acc_normalised)  # Original acc loss 40
        return acc_loss

    def compute_vel_loss(self, vel):
        vel_normalised = vel.copy()
        mean_vel = vel.mean(axis=0)
        mean_rot = 1e-2  # -1e-3
        mean_ty = 4e-4  # -4e-5
        # Normalise the velocity
        vel_normalised[:, 0] = vel_normalised[:, 0] / mean_rot
        vel_normalised[:, 1] = vel_normalised[:, 1] / mean_ty
        vel_loss = np.linalg.norm(vel_normalised)  # Original vel loss 40
        return vel_loss

    def ineq_constraint(self, x):
        """ Example of constraints from the CMA-ES implementation
        """
        rot_max = self.max_rot * self.nr_traj_points  # 2.9 deg min, total 144 deg
        ty_max = self.max_ty * self.nr_traj_points  # max on traj points is 0.002, max possible total ty 0.255
        constraints = np.array([rot_max, ty_max])
        actions = np.zeros((self.traj_points+1, 2))
        actions[1:, 0] = x[:self.traj_points] * self.scale_rot
        actions[1:, 1] = x[self.traj_points:] * self.scale_ty

        # Compute the velocity and acceleration
        vel, acc = self.compute_vel_acc(actions)

        # Compute the velocity constraint for the rotation
        vel_rot_upper = np.abs(vel) - constraints

        constraints = []
        # Denormalise so that the constraints are applied correctly
        r_constraints = vel_rot_upper[:, 0] / self.scale_rot
        t_constraints = vel_rot_upper[:, 1] / self.scale_ty
        constraints.extend(r_constraints.flatten().tolist())
        constraints.extend(t_constraints.flatten().tolist())
        # constraints.extend(vel_rot_lower.flatten().tolist())
        constraints = np.array(constraints)
        # Create an array with the
        return constraints

    def interpolate_trajectory(self, x, type_interp='pchip'):
        # Interpolate both the translation and rotation
        rot_points = [self.rx_init]  # Initially 180
        rotation = self.rx_init + x[:self.sample_traj.shape[0]] * self.scale_rot
        ty_points = [0.0]
        translation = x[self.sample_traj.shape[0]:] * self.scale_ty
        # Get the rotation and translation and scale them, shape 30x1 using 10 trajectory points
        rot_points.extend(rotation.tolist())  # radians
        ty_points.extend(translation.tolist())

        # Interpolate both the translation and rotation
        traj_idx = np.arange(0, self.horizon+1, self.nr_traj_points)  # TO-DO: can use self.traj_points
        idx_interp = np.arange(self.horizon)
        if type_interp == 'cubic':
            f_rot = interp1d(x=traj_idx, y=rot_points, kind='cubic')
            f_ty = interp1d(x=traj_idx, y=ty_points, kind='cubic')
            # Get the interpolated trajectory
            traj_rot = f_rot(idx_interp)
            traj_ty = f_ty(idx_interp)
        else:
            traj_rot = pchip_interpolate(traj_idx, rot_points, idx_interp)
            traj_ty = pchip_interpolate(traj_idx, ty_points, idx_interp)

        return traj_rot, traj_ty

    def compute_vel_noninterp(self, x):
        rot = x[:self.traj_points] * self.scale_rot
        ty = x[self.traj_points:] * self.scale_ty
        # max_rot = 28.64
        rot_max = self.max_rot * self.nr_traj_points
        # max ty on 30 traj points = 0.05
        ty_max = self.max_ty * self.nr_traj_points

        drot = rot[1:] - rot[:-1]
        dty = ty[1:] - ty[:-1]
        ineq_rot = np.abs(drot) - rot_max
        ineq_ty = np.abs(dty) - ty_max
        max_irot = ineq_rot.max()
        max_ity = ineq_ty.max()

        max_ineq = max_irot if max_irot > max_ity else max_ity

        loss_interp = np.exp(max_ineq)
        return loss_interp

    def compute_loss(self, end_position, actions, cup_states=None, coffee_states=None, x=None):
        wasserstein_loss = self.loss(end_position, self.desired_pos).item()
        vel, acc = self.compute_vel_acc(actions)
        vel_loss = self.compute_vel_loss(vel)
        acc_loss = self.compute_acc_loss(acc)

        if x is not None:  # Compute a penalty for the non-interpolated points
            interp_loss = self.compute_vel_noninterp(x)
        else:
            interp_loss = 0.0

        # Sum all the losses and apply the coefficients
        loss = self.beta * wasserstein_loss + self.alpha * vel_loss + self.gamma * acc_loss + \
               self.rho * interp_loss
        return loss, wasserstein_loss, vel_loss, acc_loss, interp_loss, 0.0


def save_loss_results(loss, initial_wass_loss, wasserstein_loss, vel_loss, acc_loss,
                      bound_loss, theta_loss, plot_dir, sim_id, time_single, time_all):
    print("Wasserstein loss (end position VS reached position)=", wasserstein_loss)
    print("wasserstein_loss=[", wasserstein_loss, "], vel_loss=[", vel_loss, "], acc_loss=[", acc_loss, "]")
    print("Acc loss=", acc_loss)
    print("Bound penalty=", bound_loss)
    print("Theta loss=", theta_loss)
    print("Total loss=", loss)
    with open(plot_dir+'/params.txt', 'a') as fd:
        text = "\n --- TEST ID" + str(sim_id) + "\n" + \
               "Initial Wasserstein loss :=[" + str(initial_wass_loss) + "]\n" + \
               "Wasserstein loss :=[" + str(wasserstein_loss) + "]\n" + \
               "Velocity loss :=["+ str(vel_loss) + "] \n" + \
               "Acceleration loss :=[" + str(acc_loss) + "]\n" + \
               "Bound loss :=[" + str(bound_loss) + "]\n" + \
               "Theta loss :=[" + str(theta_loss) + "]\n" + \
               "Total loss :=["+ str(loss) + "]\n" + \
               "Time single :=[" + str(time_single) + "]\n" + \
               "Total time :=[" + str(time_all) + "]\n"
        fd.write(text)
