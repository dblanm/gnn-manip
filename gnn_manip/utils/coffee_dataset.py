import numpy as np
import pandas as pd
import torch
import json
import os.path as osp

import torch.utils.data as tcd

from torch_geometric.data import Data, Dataset

# Module libs
from gnn_manip.utils.utils import compute_acceleration
from gnn_manip.utils.collate_utils import *

# Define some general dataset functions


def read_metadata(metadata_file):
    # Load metadata
    m_file = open(metadata_file)
    metadata = json.load(m_file)
    m_file.close()
    # Extract data_dim, sequence length, bounds and idx
    data_dim = metadata["data_dim"]
    time_steps = metadata["sequence_length"]
    metadata_bounds = torch.tensor(metadata["bounds"])
    bounds = {
        'upper_bounds': metadata_bounds[:,1],
        'lower_bounds': metadata_bounds[:,0]
    }
    cartesian_idx = metadata["cartesian_idx"]
    control_idx = metadata["control_idx"]
    material_id = metadata["material_id"]
    # Extract vel and acc mean and std
    vel_mean = torch.tensor(metadata["vel_mean"])
    vel_std = torch.tensor(metadata["vel_std"])
    acc_mean = torch.tensor(metadata["acc_mean"])
    acc_std = torch.tensor(metadata["acc_std"])
    stats = {
        'velocity_mean': vel_mean, 'velocity_std': vel_std,
        'acceleration_mean': acc_mean, 'acceleration_std': acc_std
    }
    return data_dim, time_steps, cartesian_idx, control_idx, material_id, bounds, stats


class CoffeeDataset(tcd.Dataset):
    def __init__(self, root, k, conn_r, split='train', noise=None, device=torch.device('cpu'), transform=None, use_control=False, max_neighbours=20):
        assert split in ['train', 'test']
        self.dir = root
        self.split = split
        simulation_data = np.array(pd.read_csv(f'{self.dir}{self.split}/sim_data.csv', header=None))
        self.files = [f'{self.dir}{self.split}/particles_{sim_id:06d}.csv' for (sim_id, _) in simulation_data]
        self.metadata_file = f'{self.dir}metadata.json'

        self.k = k            # Use the previous k observations as the state
        self.conn_r = conn_r  # Connectivity radius
        self.max_neighbours = max_neighbours

        self.device = device
        self.noise = noise
        self.use_control = use_control

        # Extract data dimensions, time steps, idx, bounds and stats from metadata file
        self.data_dim, self.time_steps, self.cartesian_idx, self.control_idx, self.material_id, self.bounds, self.stats = read_metadata(self.metadata_file)

        # Load data from files
        self._load_data(self.files)
        
        self.graph_attr = self._get_graph_attr()

        super(CoffeeDataset, self).__init__()

    def _load_data(self, files):
        next_pos_list, obs_list = [], []

        for file in files:
            # Read the file
            data = np.array(pd.read_csv(file, header=None)).reshape(self.time_steps, -1, self.data_dim)
            horizon = self.time_steps - self.k
            # Extract positions from data
            pos_data = data[:,:,self.cartesian_idx]

            for t in range(horizon):
                obs_idx = np.array(range(t,t+self.k))                   # Observation indices, the last k observations
                obs_seq = torch.from_numpy(data[obs_idx,:,:]).float()   # Last k observations

                next_pos = torch.from_numpy(pos_data[t+self.k]).float()

                if self.use_control:
                    # Create control inputs for the data (since dataset does not include them)
                    # We will only be considering control input of the last observation
                    # Control input in this case is next velocity for rigid body particles, other particles have control 0
                    no_control = obs_seq[:,:,self.material_id] != 1
                    ctr_input = next_pos - obs_seq[:,:,self.cartesian_idx]
                    ctr_input[no_control] = 0
                    # Add control inputs to the observations
                    obs_seq = torch.cat((obs_seq, ctr_input.float()), dim=-1)

                next_pos_list.append(next_pos)
                obs_list.append(obs_seq)

        self.raw_samples = {'observations': obs_list, 'next_positions': next_pos_list}

    def __len__(self):
        return len(self.raw_samples['observations'])

    def __getitem__(self, idx):
        obs_seq = self.raw_samples['observations'][idx]
        next_pos = self.raw_samples['next_positions'][idx]
        nodes, edge_attr, senders, receivers, tgt_acc_norm = self.graph_attr.process(obs_seq, next_pos)
        edge_index = torch.stack((senders, receivers)).long()
        data = Data(x=nodes, edge_attr=edge_attr, edge_index=edge_index, y=tgt_acc_norm)
        return data

    def _get_graph_attr(self):
        noise = self.noise
        if self.use_control:
            graph_attr = GraphBoundedMultimaterialControl(
                                    conn_r=self.conn_r, stats=self.stats, 
                                    cartesian_idx=self.cartesian_idx,
                                    material_idx=[self.material_id],
                                    control_idx= self.control_idx,
                                    bounds=self.bounds, noise=noise,
                                    max_neighbours=self.max_neighbours)
        else:
            graph_attr = GraphBoundedMultimaterial(
                                      conn_r=self.conn_r, stats=self.stats, 
                                      cartesian_idx=self.cartesian_idx,
                                      material_idx=[self.material_id],
                                      bounds=self.bounds, noise=noise,
                                      max_neighbours=self.max_neighbours)
        return graph_attr


# Many functions are highly inspired by cloth_dataset
class CoffeeTestDataset(tcd.Dataset):
    def __init__(self, directory, k, conn_r, split='train', noise=None, max_neighbours=20, device=torch.device('cpu'),
                 use_control=False, sim_id=1):
        self.dir = directory
        self.split = split
        self.files = [f'{self.dir}{self.split}/particles_{sim_id:06d}.csv']
        self.metadata_file = f'{self.dir}metadata.json'

        self.k = k            # Use the previous k observations as the state
        self.conn_r = conn_r  # Connectivity radius
        self.max_neighbours = max_neighbours

        self.use_control = use_control

        self.device = device
        self.noise = noise

        # Extract data dimensions, time steps, idx, bounds and stats from metadata file
        self.data_dim, self.time_steps, self.cartesian_idx, \
        self.control_idx, self.material_id, self.bounds, self.stats = read_metadata(self.metadata_file)

        # Load data from files
        self._load_data(self.files)
        
        self.graph_attr = self._get_graph_attr()

        super(CoffeeTestDataset, self).__init__()

    def _load_data(self, files):
        next_pos_list, obs_list = [], []

        for file in files:
            # Read the file
            data = np.array(pd.read_csv(file, header=None)).reshape(self.time_steps, -1, self.data_dim)
            horizon = self.time_steps - self.k
            # Extract positions from the data
            pos_data = data[:, :, self.cartesian_idx]

            for t in range(horizon):
                obs_idx = np.array(range(t, t+self.k))                   # Observation indices, the last k observations
                obs_seq = torch.from_numpy(data[obs_idx, :, :]).float()
                pos_seq = torch.from_numpy(pos_data[obs_idx]).float()   # Last k observed positions
                
                next_pos = torch.from_numpy(pos_data[t+self.k]).float()

                if self.use_control:
                    # Create control inputs for the data (since dataset does not include them)
                    # We will only be considering control input of the last observation
                    # Control input in this case is next velocity for rigid body particles, other particles have control 0
                    no_control = obs_seq[:,:,self.material_id] != 1
                    ctr_input = next_pos - obs_seq[:,:,self.cartesian_idx]
                    ctr_input[no_control] = 0
                    # Add control inputs to the observations
                    obs_seq = torch.cat((obs_seq, ctr_input.float()), dim=-1)

                next_pos_list.append(next_pos)
                obs_list.append(obs_seq)

        self.raw_samples = {'observations': obs_list, 'next_positions': next_pos_list}

    def __len__(self):
        return len(self.raw_samples['observations'])

    def __getitem__(self, index):
        in_state = self.raw_samples['observations'][index]
        out_state = self.raw_samples['next_positions'][index]
        return (in_state, out_state)

    def _get_graph_attr(self):
        noise = self.noise
        if self.use_control:
            graph_attr = GraphBoundedMultimaterialControl(
                                    conn_r=self.conn_r, stats=self.stats, 
                                    cartesian_idx=self.cartesian_idx,
                                    material_idx=[self.material_id],
                                    control_idx= self.control_idx,
                                    bounds=self.bounds, noise=noise,
                                    max_neighbours=self.max_neighbours)
        else:
            graph_attr = GraphBoundedMultimaterial(
                                      conn_r=self.conn_r, stats=self.stats, 
                                      cartesian_idx=self.cartesian_idx,
                                      material_idx=[self.material_id],
                                      bounds=self.bounds, noise=noise,
                                      max_neighbours=self.max_neighbours)
        return graph_attr
