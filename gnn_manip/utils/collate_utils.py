# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#

import abc
import torch

# Module libs
from gnn_manip.utils.utils import get_connectivity, get_nodes_vel, \
    get_edges_displacement, compute_acceleration, random_walk_noise


class GraphAttributes:

    def __init__(self, conn_r, stats, cartesian_idx, action_idx, bounds, noise_std):
        self.conn_r = conn_r
        self.stats = stats
        self.cartesian_idx = cartesian_idx
        self.action_idx = action_idx
        self.bounds = bounds
        self.noise_std = noise_std

    def process(self, obs, tgt):
        if self.noise_std == None:
            return self._process_simple(obs, tgt)
        else:
            return self._process_noisy(obs, tgt)

    def _process_simple(self, obs, tgt):
        last_pos = obs[-1, :, self.cartesian_idx]

        nodes = self.compute_nodes(obs)  # Compute the nodes observations

        senders, receivers = get_connectivity(last_pos, self.conn_r)  # Get the graphs connectivity

        edge_attr = self.compute_edges(obs, senders, receivers)

        nodes_tgt = self.compute_target(obs, tgt)

        return nodes, edge_attr, senders, receivers, nodes_tgt

    def _process_noisy(self, obs, tgt):
        pos_seq = obs[:, :, self.cartesian_idx]
        noise_sequence = random_walk_noise(pos_seq, self.noise_std)  # Shape (K, cloth points, pos attributes)
        # Include actions shape
        if noise_sequence.shape != obs.shape:
            shape_diff = obs.shape[2] - noise_sequence.shape[2]
            zero_tensor = torch.zeros((pos_seq.shape[0], pos_seq.shape[1], shape_diff))
            noise_sequence = torch.cat((noise_sequence, zero_tensor), dim=2)
        # Define the noisy position sequence
        noisy_obs = obs + noise_sequence

        last_pos = noisy_obs[-1, :, self.cartesian_idx]

        # We add the noise of the last position to the next position (target)
        noisy_tgt = tgt + noise_sequence[-1, :, self.cartesian_idx]

        nodes = self.compute_nodes(noisy_obs)  # Compute the nodes observations

        senders, receivers = get_connectivity(last_pos, self.conn_r)  # Get the graphs connectivity

        edge_attr = self.compute_edges(noisy_obs, senders, receivers)

        noisy_acc = self.compute_target(noisy_obs, noisy_tgt)

        return nodes, edge_attr, senders, receivers, noisy_acc

    def process_collate(self, batch):
        nodes_list, edge_attr_list, edge_index_list, tgt_list = [], [], [], []

        for i in range(len(batch)):
            obs = batch[i][0]
            tgt = batch[i][1]
            nodes, edge_attr, senders, receivers, nodes_tgt = self.process(obs, tgt)
            edge_index_batch = torch.stack((senders, receivers)).long().T
            edge_index = edge_index_batch + nodes.shape[0] * i
            nodes_list.append(nodes)
            edge_attr_list.append(edge_attr)
            edge_index_list.append(edge_index)
            tgt_list.append(nodes_tgt)

        nodes = torch.cat(nodes_list)
        edge_attr = torch.cat(edge_attr_list)
        edge_index = torch.cat(edge_index_list).T
        nodes_tgt = torch.cat(tgt_list)

        return nodes, edge_attr, edge_index, nodes_tgt

    def evaluate(self, batch):
        nodes_list, edge_attr_list, edge_index_list, node_tgt_list = [], [], [], []
        pos_seq_list, next_pos_list = [], []
        for i in range(len(batch)):
            obs = batch[i][0]
            tgt = batch[i][1]
            nodes, edge_attr, senders, receivers, nodes_tgt = self.process(obs, tgt)
            edge_index_batch = torch.stack((senders, receivers)).long().T
            edge_index = edge_index_batch + nodes.shape[0] * i
            nodes_list.append(nodes)
            edge_attr_list.append(edge_attr)
            edge_index_list.append(edge_index)
            node_tgt_list.append(nodes_tgt)
            pos_seq_list.append(obs)
            next_pos_list.append(tgt)

        nodes = torch.cat(nodes_list)
        edge_attr = torch.cat(edge_attr_list)
        edge_index = torch.cat(edge_index_list).T
        nodes_tgt = torch.cat(node_tgt_list)
        pos_seq = torch.stack(pos_seq_list)
        tgt_next_pos = torch.stack(next_pos_list)

        return nodes, edge_attr, edge_index, nodes_tgt, pos_seq, tgt_next_pos

    @abc.abstractmethod
    def compute_nodes(self, obs):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_edges(self, obs, senders, receivers):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_target(self, obs, tgt):
        raise NotImplementedError


class GraphSimple(GraphAttributes):

    def __init__(self, conn_r, stats, cartesian_idx, action_idx=None, bounds=None, noise=None):
        super(GraphSimple, self).__init__(conn_r, stats, cartesian_idx, action_idx, bounds, noise)

    def compute_nodes(self, obs):
        pos_seq = obs[:, :, self.cartesian_idx]
        last_pos = pos_seq[-1]
        # Returns flatten velocity seq (batch_size, cloth_points, vels=(vel_t-K, ..., vel_t))
        vel_attr = get_nodes_vel(pos_seq, self.stats['velocity_mean'], self.stats['velocity_std'])

        nodes = torch.cat((last_pos, vel_attr), dim=-1)

        return nodes

    def compute_edges(self, obs, senders, receivers):
        last_pos = obs[-1, :, self.cartesian_idx]
        edge_attr = get_edges_displacement(last_pos, senders, receivers, self.conn_r)

        return edge_attr

    def compute_target(self, obs, tgt):
        # Obs is the position sequence, tgt is the next_position
        pos_seq = obs[:, :, self.cartesian_idx]

        tgt_acc = compute_acceleration(tgt, pos_seq)

        # Normalise the target acceleration
        tgt_acc_norm = (tgt_acc - self.stats['acceleration_mean']) / self.stats['acceleration_std']

        return tgt_acc_norm


class GraphBoundedMultimaterial(GraphSimple):

    def __init__(self, conn_r, stats, cartesian_idx, material_idx, bounds, noise=None, max_neighbours=20):
        self.material_idx = material_idx
        self.max_neighbours = max_neighbours
        super(GraphBoundedMultimaterial, self).__init__(conn_r, stats, cartesian_idx, None, bounds, noise)
 
    # Most of this reuses GraphSimple _process_noisy() function
    # but adds noise to diffent attributes  
    def _process_noisy(self, obs, tgt):
        pos_seq = obs[:, :, self.cartesian_idx]
        noise_sequence = random_walk_noise(pos_seq, self.noise_std)  # Shape (K, cloth points, pos attributes)
        # 
        if noise_sequence.shape != obs.shape:
            zero_tensor = torch.zeros(obs.shape)
            zero_tensor[:,:,self.cartesian_idx] = noise_sequence
            noise_sequence = zero_tensor
        # Define the noisy position sequence
        noisy_obs = obs + noise_sequence

        last_pos = noisy_obs[-1, :, self.cartesian_idx]

        # We add the noise of the last position to the next position (target)
        noisy_tgt = tgt + noise_sequence[-1, :, self.cartesian_idx]

        nodes = self.compute_nodes(noisy_obs)  # Compute the nodes observations

        senders, receivers = get_connectivity(last_pos, self.conn_r, self.max_neighbours)  # Get the graphs connectivity

        edge_attr = self.compute_edges(noisy_obs, senders, receivers)

        noisy_acc = self.compute_target(noisy_obs, noisy_tgt)

        return nodes, edge_attr, senders, receivers, noisy_acc

    def compute_nodes(self, obs):
        pos_seq = obs[:,:,self.cartesian_idx]
        last_pos = pos_seq[-1]

        vel_attr = get_nodes_vel(pos_seq, self.stats['velocity_mean'], self.stats['velocity_std'])

        lower_bound_attr = last_pos - self.bounds['lower_bounds']
        upper_bound_attr = self.bounds['upper_bounds'] - last_pos
        bounds_attr = torch.clamp(torch.cat((lower_bound_attr, upper_bound_attr), dim=1)/self.conn_r, min=-1, max=1)

        material_attr = obs[-1,:,self.material_idx]
        nodes = torch.cat((vel_attr, bounds_attr, material_attr), dim=-1)

        return nodes


class GraphBoundedMultimaterialControl(GraphBoundedMultimaterial):

    def __init__(self, conn_r, stats, cartesian_idx, material_idx, control_idx, bounds, noise=None, max_neighbours=20):
        self.control_idx = control_idx
        super(GraphBoundedMultimaterialControl, self).__init__(conn_r, stats, cartesian_idx, material_idx, bounds, noise, max_neighbours=max_neighbours)

    def compute_nodes(self, obs):
        pos_seq = obs[:,:,self.cartesian_idx]
        last_pos = pos_seq[-1]
        # Get last k velocities
        vel_attr = get_nodes_vel(pos_seq, self.stats['velocity_mean'], self.stats['velocity_std'])
        # Get scaled and clipped distance to boundaries
        lower_bound_attr = last_pos - self.bounds['lower_bounds']
        upper_bound_attr = self.bounds['upper_bounds'] - last_pos
        bounds_attr = torch.clamp(torch.cat((lower_bound_attr, upper_bound_attr), dim=1)/self.conn_r, min=-1, max=1)
        # Get material
        material_attr = obs[-1,:,self.material_idx]
        # Get control inputs
        control_attr = (obs[-1,:,self.control_idx] - self.stats['velocity_mean']) / self.stats['velocity_std']
        nodes = torch.cat((vel_attr, bounds_attr, material_attr, control_attr), dim=-1)

        return nodes

