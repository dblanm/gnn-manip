import argparse
import time
import numpy as np

import torch
import torch.nn as nn

from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import sys
import os
sys.path.append(os.getcwd())

from gnn_manip.utils.coffee_dataset import CoffeeDataset
from gnn_manip.models import EncProcDecGNN


def save_model(model, filename, args):
    torch.save(model.state_dict(), filename)
    if args.print_info:
        print('Saved model...')


def load_model(model, filename, args):
    state_dict = torch.load(filename, map_location=args.device)
    model.load_state_dict(state_dict)
    if args.print_info:
        print('Loaded model...')


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def save_ckp(state, check_point_dir):
    f_path = check_point_dir / 'checkpoint.pt'
    torch.save(state, f_path)


def get_model_output(model: nn.Module, x, edge_attr, edge_index):
    return model.forward(x, edge_attr, edge_index)


def loss_batch(model: nn.Module, criterion, optimizer, batch, material_attr_id, test=False, args=None):
    x          = batch.x.to(args.device)
    edge_attr  = batch.edge_attr.to(args.device)
    edge_index = batch.edge_index.to(args.device) 
    nodes_tgt  = batch.y.to(args.device)
    # Get model prediction
    nodes_pred = get_model_output(model, x, edge_attr, edge_index)
    # Compute loss
    loss = None
    if args.use_updated_loss:
        sand_particles = x[:,material_attr_id] < 0.5    # TODO: Replace with nicer way of finding sand particles
        nodes_pred_sand_only = nodes_pred[sand_particles]
        nodes_tgt_sand_only = nodes_tgt[sand_particles]
        loss = criterion(nodes_pred_sand_only, nodes_tgt_sand_only) / nodes_tgt_sand_only.shape[0]
    else:
        loss = criterion(nodes_pred, nodes_tgt) / nodes_pred.shape[0]
    if not test:
        # Reset gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Update paramaters
        optimizer.step()
    return loss.item()


def train_epoch(dataloader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer,
                criterion, material_attr_id, args):
    total_loss = []
    model.train()
    for data in dataloader:
        total_loss.append(loss_batch(model, criterion, optimizer, data, material_attr_id, test=False, args=args))
    return total_loss


def test_epoch(dataloader: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer,
               criterion, material_attr_id, args):
    total_loss = []
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            total_loss.append(loss_batch(model, criterion, optimizer, data, material_attr_id, test=True, args=args))
    return total_loss


def train_test(train_loader: DataLoader, test_loader: DataLoader,
               dataset: CoffeeDataset, model: nn.Module, args, writer: SummaryWriter):
    avg_losses_train, avg_losses_test = [], []
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Define learning rate scheduler
    use_exponential_lr_decay = args.use_exp_lr_decay
    use_linear_lr_decay = True if args.lr_decay_final is not None and not use_exponential_lr_decay else False
    if use_linear_lr_decay:
        print('Using linear learning rate decay')
        lr_scheduler = torch.optim.swa_utils.SWALR(optimizer, anneal_strategy="linear", anneal_epochs=args.epochs, swa_lr=args.lr_decay_final)
    elif use_exponential_lr_decay:
        print('Using exponential learning rate decay')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    # Define loss function
    criterion = torch.nn.L1Loss(reduction='sum')
    # Calculate material attribute id
    material_attr_id = -1 - len(dataset.control_idx) if args.use_control else -1

    model_dir = writer.log_dir

    epochs = args.epochs
    print('Starting training...\n')
    for ep in range(epochs):
        # Train epoch
        start_time = time.time()
        train_loss = train_epoch(train_loader, model, optimizer, criterion, material_attr_id, args)
        avg_train_loss = np.mean(train_loss)
        avg_losses_train.append(avg_train_loss)
        mid_time = time.time()
        train_time = mid_time - start_time
        # Test epoch for validation loss if specified
        avg_test_loss = 0
        if test_loader is not None:
            test_loss = test_epoch(test_loader, model, optimizer, criterion, material_attr_id, args)
            avg_test_loss = np.mean(test_loss)
        avg_losses_test.append(avg_test_loss)
        test_time = time.time() - mid_time
        if args.print_info:
            print(f'Epoch [{ep+1:03d}/{epochs:03d}]: Train Loss {avg_train_loss}    ( {train_time:.2f} sec );    Test Loss {avg_test_loss}    ( {test_time:.2f} sec );    Total time: {train_time + test_time:.2f} sec')
        writer.add_scalar("Batch average train loss", avg_train_loss, ep)
        # Save the model
        if (ep+1) % args.save_freq == 0:
            save_ckp(state, check_point_dir)
            save_model(model, f'{model_dir}gns_model_{ep+1:06d}.pth', args)
            np.save(f'{model_dir}train_losses_{epochs:06d}.npy', np.array(avg_losses_train))
            if test_loader is not None:
                np.save(f'{model_dir}test_losses_{epochs:06d}.npy', np.array(avg_losses_test))
        # Decay learning rate
        if use_linear_lr_decay or use_exponential_lr_decay and ep>500:
            lr_scheduler.step()
    # Save finished model and average losses
    save_model(model, f'{model_dir}gns_model_{epochs:06d}.pth', args)
    save_model(model, f'{model_dir}gns_model_final.pth', args)
    print('\nFinished training!')
    np.save(f'{model_dir}train_losses_{epochs:06d}.npy', np.array(avg_losses_train))
    if test_loader is not None:
        np.save(f'{model_dir}test_losses_{epochs:06d}.npy', np.array(avg_losses_test))


def get_dataloader(args):
    print(f'MAX NEIGHBOURS: {args.max_neighbours}')
    # Load training dataset
    train_dataset = CoffeeDataset(
                                root=args.data_dir, k=args.k_steps, conn_r=args.conn_r, 
                                noise=args.noise_std, use_control=args.use_control,
                                max_neighbours=args.max_neighbours)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # Load test dataset if testing model while training
    test_dataset, test_loader = None, None
    if args.test_model:
        test_dataset = CoffeeDataset(
                                    root=args.data_dir, k=args.k_steps, conn_r=args.conn_r, 
                                    noise=args.noise_std, use_control=args.use_control,
                                    max_neighbours=args.max_neighbours, split='test')
        
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader


def get_model(dataloader, args):
    #node_dim, edge_dim, out_dim = (15, 3, 2) if not args.use_control else (17, 3, 2)
    node_dim, edge_dim, out_dim = None, None, None
    for batch in dataloader:
        node_dim = len(batch.x[0])
        edge_dim = len(batch.edge_attr[0])
        out_dim  = len(batch.y[0])
        break
    # print(node_dim, edge_dim, out_dim)
    gnn_model = EncProcDecGNN(node_dim=node_dim, edge_dim=edge_dim, out_dim=out_dim,
                              hidden_size=args.hidden_size, num_layers=args.num_layers, m_steps=args.message_steps).to(args.device)
    return gnn_model


def get_writer(args):
    if args.load_model is not None:
        model_folder = args.load_model.split('/')[:-1]
        writer_name = '/'.join(model_folder)
    else:
        dateTimeObj = datetime.now()
        main_folder = "runs/"
        date_test = str(dateTimeObj.month) + "m_" + str(dateTimeObj.day) + "d_" + str(dateTimeObj.hour) \
                    + "h_" + str(dateTimeObj.minute) + "m"
        lr_str = str(args.batch_size) + "B_" + str(args.lr).replace(".", "_") + "lr_"
        model_name = args.model + "_" + str(args.hidden_size) + "H_" + str(args.num_layers) + "L_"
        steps_str = str(args.k_steps) + "K_" + str(args.message_steps) + "msg_"
        writer_name = main_folder + model_name + lr_str + steps_str  + date_test
    writer = SummaryWriter(writer_name)

    return writer


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_dataset, test_dataset, train_loader, test_loader = get_dataloader(args)
    print("Data loaded")
    # Get the model
    gnn_model = get_model(train_loader, args)
    if args.load_model is not None:
        load_model(gnn_model, args.load_model, args)
    else:
        print('Start with new model...')
    # Create the writer
    writer = get_writer(args)
    # Train the model
    train_test(train_loader, test_loader, train_dataset, gnn_model, writer, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains sand dynamic model.')
    # Define datapaths
    parser.add_argument('-d', '--data_dir', help='Dataset directory')
    parser.add_argument('--model_dir', help='Directory where model files will be saved')
    parser.add_argument('--load_model', default=None, help='Specific path to model to load')
    # Define model parameters
    parser.add_argument('-c', '--use_control', action='store_true', help='Use control inputs')
    parser.add_argument('--k_steps', type=int, default=6, help='Previous k positions to be used to compute node attributes')
    parser.add_argument('--conn_r', type=float, default=0.015, help='Connectivity radius for graph formation of particles')
    parser.add_argument('--max_neighbours', type=int, default=20, help='Maximum number of neighbors for each node in graph')
    parser.add_argument('--noise_std', type=float, default=None, help='Noise std for random walk noise added to particle positions')
    parser.add_argument('--message_steps', type=int, default=10, help='Number of message passing steps (GN blocks)')
    parser.add_argument('--hidden_size', type=int, default=128, help='Size of the hidden layer in MLPs')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in MLPs')
    # Define training parameters
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate, defaults to 1e-4')
    parser.add_argument('--lr_decay_final', type=float, default=None, help="Final learning rate if using decaying learning rate, decays linearly")
    parser.add_argument('--use_exp_lr_decay', action='store_true', help='Use exponental learning rate with gamma=--gamma')
    parser.add_argument('--gamma', type=float, default=0.997, help='Gamma for exponential learning rate, default=0.997')
    parser.add_argument('--use_updated_loss', action='store_true', help='Use loss that only considers sand particles')
    # Define other arguments
    parser.add_argument('--print_info', action='store_true', help='Prints information about training progress. i.a. loss of each epoch')
    parser.add_argument('--test_model', action='store_true', help="Test the model in training at the end of each epoch")
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda:0', 'cuda:1'], help="Training device, the options are: 'cpu', 'cuda:0', 'cuda:1'")
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--save_freq', type=int, default=100, help='Save every "save_freq" epochs')
    args = parser.parse_args()
    main(args)
