from __future__ import print_function
import os
import time
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

import models 
from utils import Datasets
from utils import model_augments
from utils.params import Params

# TODO: update to augments and roc auc
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "val_json", type=str, help="Directory of validation json file which indictates the best epoch.")
    parser.add_argument(
        "eval_iter", type=int, default=5, help="Number of times to train and evaluate model")
    args = parser.parse_args()

    with open(args.val_json) as f:  
        model_params = json.load(f)

    params = Params("hparams.yaml", model_params["model"])
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=params.gpu_vis_dev
    use_gpu= torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    log_dir = os.path.join(params.log_dir, "eval_logs")
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    model_module = __import__('.'.join(['models', params.model_name]),  fromlist=['object'])
    Dataset = getattr(Datasets, params.dataset_class)
    
    acc_scores = []
    for iter_i in range(args.eval_iter):
        print("Training model for iteration {}...".format(iter_i))
        model = model_module.net().to(device)
        train = model_module.train
        test = model_module.test

        optimizer = optim.Adam(model.parameters(), lr=model_params['lr'])
        train_data_dir = os.path.join(params.data_dir, "train")
        Dataset = getattr(Datasets, params.dataset_class)

        augments_train = getattr(model_augments, params.augments_train)()
        train_data = Dataset(params.data_dir + "/train", transform=None, augmentations=augments_train)

        train_loader = DataLoader(
            train_data, 
            batch_size=params.batch_size,
            shuffle=True
            )
        if not os.path.exists(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)
        for epoch in range(1, model_params["best_val_epoch"] + 1):
            train(model, device, train_loader, optimizer)
        # Just save the last epoch of each iteration.
        torch.save(
            model.state_dict(), os.path.join(
                params.checkpoint_dir,
                "checkpoint_{}_epoch_{}_iter_{}".format(
                model_params["model"], 
                epoch,
                iter_i
                )
            )
        )
        print("Evaluating model for iteration {}...".format(iter_i))

        test_data_dir = os.path.join(params.data_dir, "test.csv")
        augments_val = getattr(model_augments, params.augments_val)()

        test_data = Dataset(params.data_dir + "/test", transform=None, augmentations=augments_val)
        test_loader = DataLoader(
            test_data, 
            batch_size=params.batch_size,
            shuffle=False
            )
        
        acc_score = test(model, device, test_loader)
        print("Accuracy for iteration {}\t {}".format(iter_i, acc_score))

        acc_scores.append(float(acc_score))
    logs ={
            "model": model_params["model"], 
            "num_epochs": model_params["best_val_epoch"],
            "lr": model_params['lr'], 
            "batch_size": model_params["batch_size"],
            "eval_iterations": args.eval_iter,
            "acc_scores": acc_scores,
            "mean_acc": float(np.mean(acc_scores)),
            "var_acc": float(np.var(acc_scores)),
            }

    with open(
        os.path.join(log_dir, "{}_{}.json".format(model_params["model"], args.eval_iter)), 'w') as f:
        json.dump(logs, f)


if __name__ == '__main__':
    main()
