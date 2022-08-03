import argparse
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import json

import geomloss

from gnn_manip.models import EncProcDecGNN
from gnn_manip.utils.coffee_dataset import CoffeeTestDataset
from gnn_manip.utils.rollout_utils import compute_rollout

from rollout_sand_dyn import get_rmse

matplotlib.use('Agg')

matplotlib.rcdefaults()

loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=.05)


def plot_wasserstein(wasserstein_distance, labels):
    nof_models = len(wasserstein_distance)
    if nof_models != len(labels):
        labels = range(nof_models)
    xticks = range(nof_models)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
    means = [np.mean(wasserstein_distance[i]) for i in range(nof_models)]
    stds  = [np.std(wasserstein_distance[i]) for i in range(nof_models)]
    medians = [np.median(wasserstein_distance[i]) for i in range(nof_models)]

    # Plot mean
    ax[0].bar(xticks, means, width=0.7)
    ax[0].set_title('Mean Wasserstein distance')

    # Plot standard deviation
    ax[1].bar(xticks, stds, width=0.7)
    ax[1].set_title('Std Wasserstein distance')

    # Plot median
    ax[2].bar(xticks, medians, width=0.7)
    ax[2].set_title('Median Wasserstein distance')

    # Add labels
    for i in range(3):
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(labels)
    np.save('was_dist.npy', np.concatenate([means, stds, medians], axis=0))
    plt.savefig('was_dist.png')

    was_dist = [np.mean(wasserstein_distance[i], axis=1) for i in range(nof_models)] # <- average timesteps for each sim
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18,12))
    stats = [[],[]]
    for i in range(nof_models):
        stats0 = {
            'med': np.median(wasserstein_distance[i]),
            'q1': np.quantile(wasserstein_distance[i], 0.25),
            'q3': np.quantile(wasserstein_distance[i], 0.75),
            'whislo': np.min(wasserstein_distance[i]),
            'whishi': np.max(wasserstein_distance[i]),
            'mean': np.mean(wasserstein_distance[i]),
            'label': labels[i]
        }
        stats1 = {
            'med': np.median(was_dist[i]),
            'q1': np.quantile(was_dist[i], 0.25),
            'q3': np.quantile(was_dist[i], 0.75),
            'whislo': np.min(was_dist[i]),
            'whishi': np.max(was_dist[i]),
            'mean': np.mean(was_dist[i]),
            'label': labels[i]
        }
        stats[0].append(stats0)
        stats[1].append(stats1)
    ax[0].bxp(stats[0], showfliers=False, showmeans=True, meanline=True)
    ax[0].set_title('Over all timesteps on all simulations')

    ax[1].bxp(stats[0], showfliers=False, showmeans=True, meanline=True)
    ax[1].set_title('Over all timesteps on all simulations logarithmic scale')
    ax[1].set_yscale('log')
    plt.savefig('bxp_wasserstein_distance.png')
    with open('bxp_wasser.json', 'w') as fp:
        json.dump(stats[0], fp)


def plot_rmses(rmses, labels):
    rmse_means = np.mean(rmses, axis=1)
    rmse_stds = np.std(rmses, axis=1)
    rmse_medians = np.median(rmses, axis=1)
    nof_values = len(rmses)
    nof_models = nof_values // 4
    xticks = range(nof_models)
    if len(labels) < nof_models:
        labels = range(nof_models)
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18,18))
    pos_idx  = range(1,nof_values,4)
    acc_idx  = range(2,nof_values,4)
    loss_idx = range(3,nof_values,4)

    # Plot mean
    ax[0,0].bar(xticks, rmse_means[pos_idx], width=0.7)
    ax[0,0].set_title('Position mean RMSE')
    ax[0,1].bar(xticks, rmse_means[acc_idx], width=0.7)
    ax[0,1].set_title('Acceleration mean RMSE')
    ax[0,2].bar(xticks, rmse_means[loss_idx], width=0.7)
    ax[0,2].set_title('Mean Wasserstein distance')

    # Plot standard deviation
    ax[1,0].bar(xticks, rmse_stds[pos_idx], width=0.7)
    ax[1,0].set_title('Position std RMSE')
    ax[1,1].bar(xticks, rmse_stds[acc_idx], width=0.7)
    ax[1,1].set_title('Acceleration std RMSE')
    ax[1,2].bar(xticks, rmse_stds[loss_idx], width=0.7)
    ax[1,2].set_title('Std Wasserstein distance')

    # Plot median
    ax[2,0].bar(xticks, rmse_medians[pos_idx], width=0.7)
    ax[2,0].set_title('Position median RMSE')
    ax[2,1].bar(xticks, rmse_medians[acc_idx], width=0.7)
    ax[2,1].set_title('Acceleration median RMSE')
    ax[2,2].bar(xticks, rmse_medians[loss_idx], width=0.7)
    ax[2,2].set_title('Median Wasserstein distance')

    # Add labels
    for i in range(3):
        for j in range(3):
            ax[i,j].set_xticks(xticks)
            ax[i,j].set_xticklabels(labels)

    plt.savefig('rmse_plot.png')


def get_model(dataset, model, m_steps, args):
    node_dim, edge_dim, out_dim = None, None, None
    for obs, tgt in dataset:
        nodes, edge_attr, _, _, acc = dataset.graph_attr.process(obs, tgt)
        node_dim = len(nodes[0])
        edge_dim = len(edge_attr[0])
        out_dim  = len(acc[0])
        break
    print(node_dim, edge_dim, out_dim)
    # Initialize model
    gnn_model = EncProcDecGNN(node_dim=node_dim, edge_dim=edge_dim, out_dim=out_dim,
                              hidden_size=128, num_layers=2, m_steps=m_steps).to(args.device)
    # Load model
    state_dict = torch.load(model, map_location=args.device)
    gnn_model.load_state_dict(state_dict)
    print('Loaded model...')
    return gnn_model


def main(args):
    assert len(args.models) == len(args.use_control), 'Number of given control definitions (--use_control) does not correspond to number of given models (--models)'
    assert len(args.message_steps) == len(args.models), 'Number of given message passing step values (--message_steps) does not correspond to number of given models (--models)'
    assert len(args.k_steps) == len(args.models), 'Number of given history length values (--k_steps) does not correspond to number of given models (--models)'
    nof_simulations = args.nof_sims
    nof_models = len(args.models)
    rmses = np.zeros((nof_models*4, nof_simulations))
    was_dist = []
    for model_idx, model in enumerate(args.models):
        # print(model)
        fig, ax = plt.subplots(figsize=(12,6))
        # Get dataset
        test_dataset = CoffeeTestDataset(directory=args.dir, split='test', k=args.k_steps[model_idx], conn_r=0.015,
                                         noise=None, use_control=args.use_control[model_idx], max_neighbours=args.max_neighbours)
        # Get and load the model
        gnn_model = get_model(test_dataset, model, args.message_steps[model_idx], args)

        # Generate rollout
        sinkhorn_losses = np.zeros((nof_simulations, test_dataset.time_steps - test_dataset.k))
        for sim_idx in range(1, 1+nof_simulations):
            test_dataset = CoffeeTestDataset(directory=args.dir, split='test', k=args.k_steps[model_idx], conn_r=0.015,
                                             noise=None, use_control=args.use_control[model_idx], sim_id=sim_idx,
                                             max_neighbours=args.max_neighbours)
            groundtruth, prediction, groundtruth_acc, prediction_acc = compute_rollout(test_dataset, gnn_model, args)

            if args.save_predictions:
                if model_idx == 0:
                    np.save(f'groundtruth_{sim_idx:03d}.npy', groundtruth[:,:,:5])
                np.save(f'prediction_{args.labels[model_idx]}_{sim_idx:03d}.npy', prediction[:,:,:5])

            coffee_particles = test_dataset[0][0][0,:,test_dataset.material_id] == 0
            rmse = get_rmse(groundtruth, prediction)
            rmse_coffee = get_rmse(groundtruth[:,coffee_particles,:], prediction[:,coffee_particles,:])
            rmse_acc = get_rmse(groundtruth_acc[:,coffee_particles,:], prediction_acc[:,coffee_particles,:],
                                cartesian_idx=[0,1,2])
            sinkhorn_loss = [loss(torch.from_numpy(groundtruth[i, coffee_particles, :][:, test_dataset.cartesian_idx]),
                                  torch.from_numpy(
                                      prediction[i, coffee_particles, :][:, test_dataset.cartesian_idx])
                                  ).item() for i in range(len(groundtruth))]

            rmses[model_idx*4 + 0, sim_idx - 1] = rmse
            rmses[model_idx*4 + 1, sim_idx - 1] = rmse_coffee
            rmses[model_idx*4 + 2, sim_idx - 1] = rmse_acc
            rmses[model_idx*4 + 3, sim_idx - 1] = np.mean(np.array(sinkhorn_loss))

            sinkhorn_losses[sim_idx-1] = np.array(sinkhorn_loss)
            ax.plot(sinkhorn_loss, label=sim_idx)
            np.save('rmses.npy', rmses)
        was_dist.append(sinkhorn_losses)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('W')
        ax.legend()
        plt.savefig(f'{args.labels[model_idx]}-{model_idx:03d}_loss.png')
    plot_rmses(rmses, args.labels)
    plot_wasserstein(was_dist, args.labels)
    np.save('rmses.npy', rmses)


if __name__ == '__main__':
    """ Usage: python plot_rmses.py -d '../tmp/datasets/coffee/' -m 'c_ul.pth' 'c.pth' 'ul.pth' --labels 'c_ul' 'c' 'ul'
        -c 1 1 0 --device 'cuda:0' --nof_sims 8 --message_steps 10 10 10 --k_steps 6 6 6
    """
    parser = argparse.ArgumentParser(description='Plot bar graph of mean RMSE of given models and test range')
    parser.add_argument('-c', '--use_control', nargs='+', type=int, required=True,
                        help='use control inputs (1=use, 0=do not use), give one for each model to be tested')
    parser.add_argument('-pr', '--predict_rigids', action='store_true',
                        help='use predicted positions for rigid body particles')
    parser.add_argument('-d', '--dir', required=True, help='dataset directory')
    parser.add_argument('--k_steps', type=int, nargs='+', required=True,
                        help="how long history is included (positions), give one for each model")
    parser.add_argument('-m', '--models', nargs='+', required=True,
                        help='path to model files to be used for rollout generation')
    parser.add_argument('--message_steps', type=int, nargs='+', required=True,
                        help="number of message passing steps, give one for each model to be tested")
    parser.add_argument('--labels', nargs='*', help='names of models to be put in plot')
    parser.add_argument('--nof_sims', type=int, default=1, help='number of simulations in test dataset')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda:0', 'cuda:1'],
                        help="Training device, the options are: 'cpu', 'cuda:0', 'cuda:1'")
    parser.add_argument('--max_neighbours', default=20, type=int, help='Number of maximum neighbours')
    parser.add_argument('--save_predictions', action='store_true', help='Save prediction numpy arrays')
    args = parser.parse_args()

    print('Using device:', args.device)
    main(args)
