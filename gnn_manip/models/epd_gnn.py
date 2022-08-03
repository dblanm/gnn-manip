# Copyright 2021 (c) Aalto University - All Rights Reserved
# Author: David Blanco Mulero <david.blancomulero@aalto.fi>
#

import torch
import torch.nn as nn
from torch_graphnet import InteractionNetwork, GraphIndependent
from gnn_manip.models.action_networks import ActionInteractionNetwork, ActionGraphIndependent


class EncProcDecGNN(nn.Module):

    def __init__(self, node_dim, edge_dim, out_dim,
                 hidden_size, num_layers, m_steps, norm_type='LayerNorm'):
        """

        Args:
            node_dim: nodes attributes dimension
            edge_dim: edge attributes dimension
            out_dim: output dimension
            hidden_size: MLPs hidden size
            num_layers: number of hidden layers
            m_steps: k message passing steps
        """
        super(EncProcDecGNN, self).__init__()
        assert (num_layers >= 2), "The number of layers num_layers must be at least 2"
        assert (m_steps >= 1), "The number of m_steps message pasting steps must be at least 1"
        self._get_layer_norm(norm_type)
        # Create the phi edge and phi node for encoder, processor and decoder
        self.encoder = GraphIndependent(phi_edge=self._build_mlp(edge_dim, hidden_size,
                                                                 hidden_size, num_layers, norm=True),
                                        phi_node=self._build_mlp(node_dim, hidden_size,
                                                                 hidden_size, num_layers, norm=True))
        # The processor takes as input the hidden size from the encoder
        # Interaction Network phi edge dimension  = edge attr + node feat + node feat
        # phi node dimension = node attr + edge feat + edge feat
        dim_phi_edge_in = 3*hidden_size
        dim_phi_node_in = 2*hidden_size

        processor_modules = []
        for i in range(m_steps):
            processor_modules.append(InteractionNetwork(phi_edge=self._build_mlp(dim_phi_edge_in, hidden_size,
                                                                                 hidden_size, num_layers, norm=True),
                                                        phi_node=self._build_mlp(dim_phi_node_in, hidden_size,
                                                                                 hidden_size, num_layers, norm=True)))
        self.processor = nn.ModuleList(processor_modules)

        # The decoder takes as input the hidden size nodes and outputs the output dimension
        self.decoder = self._build_mlp(hidden_size, hidden_size, out_dim, num_layers, norm=False)

    def _get_layer_norm(self, norm_type):

        if norm_type == 'BatchNorm':
            print("Using BatchNorm")
            self.norm_layer = lambda x: nn.BatchNorm2d(x)
        elif norm_type == 'InstanceNorm':
            print("Using InstanceNorm")
            self.norm_layer = lambda  x: nn.InstanceNorm2d(x)
        else:
            print("Using LayerNorm")
            self.norm_layer = lambda x: nn.LayerNorm(x)

    def print_structure(self):
        print("Encoder ")
        print(self.encoder)
        print("Processor")
        for module in self.processor:
            print(module)
        print("Decoder ")
        print(self.decoder)

    def _build_mlp(self, input_dim, hidden_size, output_dim, num_layers, norm=False):

        modules = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for i in range(num_layers-1):  # First layer is already defined
            modules.append(nn.Linear(hidden_size, hidden_size))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_size, output_dim))

        if norm:
            modules.append(self.norm_layer(output_dim))

        model = nn.Sequential(*modules)
        return model

    def forward(self, nodes, edge_attr, edge_index):
        # Encode the input nodes and edges
        latent_node_0, latent_edge_0, _ = self.encoder(nodes, edge_attr, edge_index)
        prev_latent_node_k = latent_node_0
        prev_latent_edge_k = latent_edge_0
        # Process the latent nodes and edges
        for in_module in self.processor:
            prev_latent_node_k, prev_latent_edge_k = self._process(in_module, prev_latent_node_k,
                                                                   prev_latent_edge_k, edge_index)
        # Decode the k-latent nodes
        decoded_nodes = self.decoder(prev_latent_node_k)

        return decoded_nodes

    def _process(self, in_module, prev_latent_node, prev_latent_edge, edge_index):
        latent_node_k, latent_edge_k, _ = in_module(prev_latent_node, prev_latent_edge, edge_index)
        # Add residuals
        output_node = latent_node_k + prev_latent_node
        output_edge = latent_edge_k + prev_latent_edge
        return output_node, output_edge
