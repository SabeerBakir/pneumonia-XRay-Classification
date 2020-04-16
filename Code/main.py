from __future__ import print_function
import os
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import models
from utils import Datasets
from utils.params import Params
from utils import model_transforms
from utils.plotting import plot_training


def main():
    start_time = time.strftime("%d%m%y_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", 
        type=str, 
        help="Pass name of model as defined in hparams.yaml."
        )
    parser.add_argument(
        "--write_data",
        required = False,
        default=False,
                help="Set to true to write_data."
        )
    args = parser.parse_args()
    # Parse our YAML file which has our model parameters. 
    params = Params("hparams.yaml", args.model_name)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=params.gpu_vis_dev
    # Check if a GPU is available and use it if so. 
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    
    # Load model that has been chosen via the command line arguments. 
    model_module = __import__('.'.join(['models', params.model_name]),  fromlist=['object'])
    model = model_module.net(**params.net_args)
    # Send the model to the chosen device. 
    # To use multiple GPUs
    # model = nn.DataParallel(model)
    model.to(device)
    # Grap your training and validation functions for your network.
    train = model_module.train
    val = model_module.val
    
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    # Write data if specified in command line arguments. 
    if args.write_data:
        data = pd.read_csv('fashionmnist/fashion-mnist_train.csv')
        test_data = pd.read_csv('fashionmnist/fashion-mnist_test.csv')
        val_split = round(data.shape[0]*0.2)
        data = shuffle(data)
        train_data = data.iloc[val_split:]
        val_data = data.iloc[:val_split]
        train_data.to_csv(os.path.join(params.data_dir, "train.csv"), index=False)
        val_data.to_csv(os.path.join(params.data_dir, "val.csv"), index=False)
        test_data.to_csv(os.path.join(params.data_dir, "test.csv"), index=False)

    # This is useful if you have multiple custom datasets defined. 
    Dataset = getattr(Datasets, params.dataset_class)

    transf_train = getattr(model_transforms, params.transforms_train)()
    transf_val = getattr(model_transforms, params.transforms_val)()

    train_data = Dataset(params.data_dir + "/train", transform=transf_train)
    val_data = Dataset(params.data_dir + "/val", transform=transf_val)

    train_loader = DataLoader(
        train_data, 
        batch_size=params.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=params.batch_size,
        shuffle=False
    )
    os.makedirs(params.log_dir, exist_ok=True)
    os.makedirs(params.checkpoint_dir, exist_ok=True)
    os.makedirs(params.data_dir, exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    val_roc_aucs = []
    val_losses = []
    train_losses = []
    train_roc_aucs = []
    for epoch in range(1, params.num_epochs + 1):
        print("Epoch: {}".format(epoch))
        # Call training function. 
        train(model, device, train_loader, optimizer)
        # Evaluate on both the training and validation set. 
        train_loss, train_roc_auc = val(model, device, train_loader)
        val_loss, val_roc_auc = val(model, device, val_loader)
        # Collect some data for logging purposes. 
        train_losses.append(float(train_loss))
        train_roc_aucs.append(train_roc_auc)
        val_losses.append(float(val_loss))
        val_roc_aucs.append(val_roc_auc)

        print('\n\ttrain Loss: {:.6f}\ttrain roc_auc: {:.6f} \n\tval Loss: {:.6f}\tval roc_auc: {:.6f}'.format(train_loss, train_roc_auc, val_loss, val_roc_auc))
        # Here is a simply plot for monitoring training. 
        # Clear plot each epoch 
        fig = plot_training(train_losses, train_roc_aucs, val_losses, val_roc_aucs)
        fig.savefig(os.path.join("figs", "{}_training_vis".format(args.model_name)))
        # Save model every few epochs (or even more often if you have the disk space).
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(params.checkpoint_dir, "checkpoint_{}_epoch_{}".
                                                        format(args.model_name, epoch)))
    # Some log information to help you keep track of your model information. 
    logs ={
        "model": args.model_name,
        "net_args": params.net_args,
        "train_losses": train_losses,
        "train_roc_aucs": train_roc_aucs,
        "val_losses": val_losses,
        "val_roc_aucs": val_roc_aucs,
        "best_val_epoch": int(np.argmax(val_roc_aucs)+1),
        "model": args.model_name,
        "lr": params.lr,
        "batch_size": params.batch_size,
        "transforms_train": str(transf_train),
        "transforms_val": str(transf_val)
    }

    with open(os.path.join(params.log_dir,"{}_{}.json".format(args.model_name,  start_time)), 'w') as f:
        json.dump(logs, f)


if __name__ == '__main__':
    main()
