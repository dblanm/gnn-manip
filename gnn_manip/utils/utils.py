# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#

import torch
import numpy as np
from sklearn.neighbors import KDTree


@torch.jit.script
def compute_acceleration(next_pos:torch.Tensor, pos_seq:torch.Tensor):
    """

    Args:
        next_pos: (N points, Cartesian dims)
        pos_seq: (timepoints, N points, Cartesian dims)
            The timepoints are provided as [t-K, ..., t]

    Returns:
        Target acceleration of points with shape (N points, Cartesian dims)
    """

    tgt_acc = next_pos - 2*pos_seq[-1, :, :] + pos_seq[-2, :, :]
    return tgt_acc


@torch.jit.script
def get_nodes_vel(pos_seq:torch.Tensor, velocity_mean: torch.Tensor, velocity_std: torch.Tensor):
    # Compute the first order diff over the position sequence (dim 0) on the last dim and normalise it
    vel_seq = torch.diff(pos_seq, n=1, dim=0)  # (k, cloth_points, vels)


    vel_seq_norm = (vel_seq - velocity_mean) / velocity_std
    # Before flattening let's permute so that flattening is performed correctly
    permuted_vel = vel_seq_norm.permute(1, 0, 2)  # From (k, cloth points, vels) -> (cloth points, k, vels)

    # Flatten the velocity seq (batch_size, cloth_points, vels=(vel_t-K, ..., vel_t))
    flat_vel_seq = torch.reshape(permuted_vel, (vel_seq.shape[1],vel_seq.shape[0]*vel_seq.shape[2]))

    return flat_vel_seq


@torch.jit.script
def get_edges_displacement(last_pos:torch.Tensor, senders:torch.Tensor,
                           receivers:torch.Tensor, conn_r:float):
    """
    Compute the edges. Since the senders & receivers size is dynamic the input is a list.
    The output edge attributes is also a list of edge attributes per graph in the batch.
    """

    # Index over the dimension cloth_points of the batch i
    pos_send = torch.index_select(last_pos, dim=0, index=senders)
    pos_recv = torch.index_select(last_pos, dim=0, index=receivers)

    # Compute the edge attributes
    relative_displacement = (pos_send - pos_recv)/conn_r
    relative_dist = torch.norm(relative_displacement, dim=-1, keepdim=True)

    edge_attr = torch.cat((relative_displacement, relative_dist), dim=-1)

    return edge_attr


def get_connectivity(pos_nodes, conn_r, max_neighbours=20):
    """
    Compute the connectivity in each graph. Since it can change per graph we return
    lists of senders and receivers.
    Args:
        pos_nodes: last position of the nodes in a batch (nodes, attributes)
        conn_r: connectivity radius to define the connectivity

    Returns: senders, receivers
    """
    # Compute the connectivity of all the nodes in a single graph
    n_nodes = pos_nodes.shape[0]
    tree = KDTree(pos_nodes)
    # Compute the receivers as the nodes neighbours indices based on conn_r
    full_receivers_list, rcv_distance = tree.query_radius(pos_nodes, conn_r, return_distance=True, sort_results=True)
    # Limit the amount of neighbours
    receivers = []
    for r in full_receivers_list:
        if len(r) >= max_neighbours:
            receivers.append(r[:max_neighbours])
        else:
            receivers.append(r)
    # Repeat a node as many times as length in the receivers
    senders_npy = np.repeat(range(n_nodes), [len(r) for r in receivers])
    senders = torch.from_numpy(senders_npy).long()
    # Append the senders and receivers
    receivers_npy = np.concatenate(receivers, axis=0)
    receivers = torch.from_numpy(receivers_npy).long()

    return senders, receivers


def random_walk_noise(pos_seq:torch.Tensor, noise_std: float):
    # We want the noise scale in the velocity at the last step to be fixed.
    # std_last_step**2 = num_velocities * std_each_step**2
    # so to keep `std_last_step` fixed, we apply at each step:
    # std_each_step `std_last_step / np.sqrt(num_input_velocities)`
    vel_seq = torch.diff(pos_seq, n=1, dim=0).float()
    noise = noise_std / vel_seq.shape[0]**0.5  #Num of velocities
    noise_dist = torch.distributions.Normal(loc=torch.zeros_like(vel_seq),
                                            scale=noise)
    noise_sample = noise_dist.sample().float()
    # Create the random walk noise in the velocity
    noisy_vel = torch.cumsum(noise_sample, dim=0)

    # Create the random walk noise in the integration
    noisy_pos = torch.cumsum(noisy_vel, dim=0)
    zero_pos = torch.zeros((1, noisy_pos.shape[1], noisy_pos.shape[2]))
    # Create the position noise with zero noise in the first position
    noise_sequence = torch.cat((zero_pos, noisy_pos), axis=0)

    return noise_sequence




