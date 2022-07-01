import argparse
import json
import os
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from tools.img_dataset import MultiviewImgDataset, SingleImgDataset
from tools.trainer import ModelNetTrainer
from models.MVCNN import MVCNN, SVCNN

parser = argparse.ArgumentParser()
parser.add_argument(
    "-name", "--name", type=str, help="Name of the experiment", default="MVCNN"
)
parser.add_argument(
    "-bs", "--batchSize", type=int, help="Batch size for the second stage", default=4
)  # it will be *12 images in each batch for mvcnn
parser.add_argument(
    "-num_models", type=int, help="number of models per class", default=1000
)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest="no_pretraining", action="store_true")
parser.add_argument(
    "-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11"
)
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument(
    "-train_path", type=str, default="data/modelnet40_images_new_12x/*/train"
)
parser.add_argument(
    "-val_path", type=str, default="data/modelnet40_images_new_12x/*/test"
)
parser.add_argument(
    "-epochs_stage1", type=int, default=1
)
parser.add_argument(
    "-epochs_stage2", type=int, default=1
)
parser.add_argument(
    "-use_pretrained_svcnn", type=bool, default=False
)
parser.set_defaults(train=False)


def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        raise ValueError(
            "ERROR: Please change the name of the run model to avoid duplication"
        )
        # print("WARNING: summary folder already exists!! It will be overwritten!!")
        # shutil.rmtree(log_dir)
        # os.mkdir(log_dir)


if __name__ == "__main__":
    args = parser.parse_args()

    pretraining = not args.no_pretraining
    log_dir = f"runs/{args.name}"
    create_folder(log_dir)
    config_f = open(os.path.join(log_dir, "config.json"), "w")
    json.dump(vars(args), config_f)
    config_f.close()

    n_models_train = args.num_models * args.num_views

    if args.use_pretrained_svcnn:
        # Load SVCNN
        cnet = SVCNN(
            "svcnn_resnet18",
            nclasses=40,
            pretraining=pretraining,
            cnn_name=args.cnn_name
        )

        path = "runs/svcnn_resnet18/stage_1"
        cnet.load(path)
    else:
        wandb.init(
            project="MVCNN Transfer Test",
            entity="icheler-team",
            name=f"{args.name}_stage_1",
            tags=["stage1"],
            config={
                "name": args.name,
                "num_models": args.num_models,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "no_pretraining": args.no_pretraining,
                "cnn_name": args.cnn_name,
                "num_views": args.num_views,
            },
        )
        # # STAGE 1
        log_dir = f"runs/{args.name}/stage_1"
        create_folder(log_dir)
        cnet = SVCNN(
            args.name, nclasses=40, pretraining=pretraining, cnn_name=args.cnn_name
        )

        optimizer = optim.Adam(
            cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        train_dataset = SingleImgDataset(
            args.train_path,
            scale_aug=False,
            rot_aug=False,
            num_models=n_models_train,
            num_views=args.num_views,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=4
        )

        val_dataset = SingleImgDataset(
            args.val_path, scale_aug=False, rot_aug=False, test_mode=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=64, shuffle=False, num_workers=4
        )
        print("num_train_files: " + str(len(train_dataset.filepaths)))
        print("num_val_files: " + str(len(val_dataset.filepaths)))
        trainer = ModelNetTrainer(
            cnet,
            train_loader,
            val_loader,
            optimizer,
            nn.CrossEntropyLoss(),
            "svcnn",
            log_dir,
            num_views=1,
        )
        print("Stage 1")
        torch.cuda.empty_cache()
        trainer.train(args.epochs_stage1)
        wandb.finish()

    # STAGE 2

    wandb.init(
        project="MVCNN Transfer Test",
        entity="icheler-team",
        name=f"{args.name}_stage_2",
        tags=["stage2"],
        config={
            "name": args.name,
            "num_models": args.num_models,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "no_pretraining": args.no_pretraining,
            "cnn_name": args.cnn_name,
            "num_views": args.num_views,
        },
    )
    log_dir = f"runs/{args.name}/stage_2"
    create_folder(log_dir)
    cnet_2 = MVCNN(
        args.name, cnet, nclasses=40, cnn_name=args.cnn_name, num_views=args.num_views
    )
    del cnet

    # Freeze All layers
    for parameters in cnet_2.parameters():
        parameters.requires_grad = False

    # Add new last layer
    if cnet_2.use_resnet:
        in_ftrs = cnet_2.net_2.in_features
        out_ftrs = cnet_2.net_2.out_features
        cnet_2.net_2 = nn.Linear(in_ftrs, out_ftrs)
    else:
        in_ftrs = cnet_2.net_2._modules["6"].in_features
        out_ftrs = cnet_2.net_2._modules["6"].out_features
        cnet_2.net_2._modules["6"] = nn.Linear(in_ftrs, out_ftrs)

    for params in cnet_2.parameters():
        print(params.requires_grad)

    optimizer = optim.Adam(
        cnet_2.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    train_dataset = MultiviewImgDataset(
        args.train_path,
        scale_aug=False,
        rot_aug=False,
        num_models=n_models_train,
        num_views=args.num_views,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4
    )  # shuffle needs to be false! it's done within the trainer

    val_dataset = MultiviewImgDataset(
        args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=4
    )
    print("num_train_files: " + str(len(train_dataset.filepaths)))
    print("num_val_files: " + str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(
        cnet_2,
        train_loader,
        val_loader,
        optimizer,
        nn.CrossEntropyLoss(),
        "mvcnn",
        log_dir,
        num_views=args.num_views,
    )
    print("Stage 2")
    torch.cuda.empty_cache()
    trainer.train(args.epochs_stage2)
    wandb.finish()
