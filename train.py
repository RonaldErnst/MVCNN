# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import wandb


parser = argparse.ArgumentParser()
parser.add_argument(
    "-name", "--name", type=str, help="Name of the experiment", default="Pretrain Test"
)
parser.add_argument(
    "-bs", "--batchSize", type=int, help="Batch size for the second stage", default=32
)
parser.add_argument(
    "-num_epochs", "--num_epochs", type=int, help="Number of epochs to train", default=4
)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument(
    "-cnn_name", "--cnn_name", type=str, help="cnn model name", default="inception"
)

def train_model(model_name, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', "val"]:
            model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            step = 0
            print(phase, len(dataloaders[phase]))
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    if model_name == "inception":
                        outputs = outputs.logits

                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    results = preds == labels
                    correct_points = torch.sum(results.long())
                    acc = correct_points.float() / results.size()[0]

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                wandb.log(
                    {
                        phase: {
                            "epoch": epoch + 1,
                            "step": step+1,
                            f"{phase}_loss": loss,
                            f"{phase}_acc": acc,
                        }
                    }
                )

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                step = step +1

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.cnn_name == "inception":
        model_conv = torchvision.models.inception_v3(weights = torchvision.models.Inception_V3_Weights.DEFAULT)
        for param in model_conv.parameters():
            param.requires_grad = False

        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, 100)

        model_params = model_conv.fc.parameters()

        img_input_size = 299
    else: 
        model_conv = torchvision.models.convnext_base(weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT)
        for param in model_conv.parameters():
            param.requires_grad = False

        model_conv.classifier._modules["2"] = nn.Linear(1024, 100)

        model_params = model_conv.classifier.parameters()
        img_input_size = 224


    training_data = datasets.CIFAR100(
        root="data",
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_input_size,img_input_size))
        ])
    )

    val_data = datasets.CIFAR100(
        root="data",
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_input_size,img_input_size))
        ])
    )

    optimizer_conv = optim.Adam(model_params, lr=args.lr, weight_decay=args.weight_decay)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    dataloaders = {
        "train": DataLoader(training_data, batch_size=64, shuffle=True, num_workers=4),
        "val": DataLoader(val_data, batch_size=64, shuffle=True, num_workers=4)
    }

    dataset_sizes = {
        "train": len(training_data),
        "val": len(val_data)
    }

    class_names = training_data.classes

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    wandb.init(
        project="pretrain-test",
        entity="icheler-team",
        name=f"{args.name}_{args.cnn_name}",
        config={
            "name": f"{args.cnn_name} Transfer Learning Test",
            "lr": args.lr,
            "weight-decay": args.weight_decay
        },
    )

    model_conv = train_model(args.cnn_name, model_conv, criterion, optimizer_conv,
                            exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=args.num_epochs)
                            
    wandb.finish()