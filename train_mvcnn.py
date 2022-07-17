import argparse
import json
import os
import wandb
import sys

# Only for windows
# Set Max Split size because it keeps crashing
if os.name == 'nt':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

import torch
import torch.nn as nn
import torch.optim as optim
from tools.img_dataset import MultiviewImgDataset, SingleImgDataset
from tools.trainer import ModelNetTrainer
from models.MVCNN import MVCNN, SVCNN

parser = argparse.ArgumentParser()
parser.add_argument(
    "-name", "--name", type=str, help="Name of the experiment",
    required=True
)
parser.add_argument(
    "-bs", "--batchSize", type=int, help="Batch size for the second stage", default=4
)  # it will be *12 images in each batch for mvcnn
parser.add_argument(
    "-num_models", type=int, help="number of models per class", default=1000
)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.01)
parser.add_argument("-no_pretraining", dest="no_pretraining", action="store_true")
parser.add_argument(
    "-cnn_name", "--cnn_name", type=str, help="cnn model name",
    required=True
)
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument(
    "-train_path", type=str, default="data/modelnet40_images_new_12x/*/train"
)
parser.add_argument(
    "-val_path", type=str, default="data/modelnet40_images_new_12x/*/test"
)
parser.add_argument(
    "-num_workers", type=int, default=4
)
parser.add_argument(
    "-num_epochs", type=int, default=1
)
parser.add_argument("-stage", type=int, required=True, help="Stage 1 or Stage 2")
parser.add_argument("-svcnn_name", type=str, default="")
parser.add_argument("-freeze", type=bool, default=True)
parser.add_argument("-continue_training", type=bool, default=False)
parser.set_defaults(train=False)


def create_folder(log_dir, continue_training=False):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    elif not continue_training:  # Dont throw error when continue training
        raise ValueError(
            "ERROR: Please change the name of the run model to avoid duplication"
        )
        # print("WARNING: summary folder already exists!! It will be overwritten!!")
        # shutil.rmtree(log_dir)
        # os.mkdir(log_dir)


if __name__ == "__main__":
    project_name = 'r-convnext-modelnet'

    args = parser.parse_args()

    n_models_train = args.num_models * args.num_views

    if not torch.cuda.is_available():
        print("Not using cuda... exiting")
        sys.exit()

    if args.stage == 1:
        wandb.init(
            project=project_name,
            entity="icheler-team",
            resume=args.continue_training,
            name=f"{args.name}_stage_1",
            tags=["stage1"],
            config={
                "name": args.name,
                "num_models": args.num_models,
                "lr": args.lr,
                "batch_size": args.batchSize,
                "weight_decay": args.weight_decay,
                "no_pretraining": args.no_pretraining,
                "cnn_name": args.cnn_name,
                "num_views": args.num_views,
            }
        )

        pretraining = not args.no_pretraining
        log_dir = f"runs/{args.name}"
        create_folder(log_dir, args.continue_training)
        config_f = open(os.path.join(log_dir, "config.json"), "w")
        json.dump(vars(args), config_f)
        config_f.close()

        # STAGE 1
        log_dir = f"runs/{args.name}/stage_1"
        create_folder(log_dir, args.continue_training)
        cnet = SVCNN(
            args.name,
            nclasses=40,
            pretraining=pretraining,
            cnn_name=args.cnn_name
        )

        optimizer = optim.Adam(
            cnet.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        train_dataset = SingleImgDataset(
            args.train_path,
            scale_aug=False,
            rot_aug=False,
            num_models=n_models_train,
            num_views=args.num_views,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchSize,
            shuffle=True,
            num_workers=args.num_workers
        )

        val_dataset = SingleImgDataset(
            args.val_path,
            scale_aug=False,
            rot_aug=False,
            test_mode=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=args.num_workers
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
        trainer.train(args.num_epochs)
        wandb.finish()

    elif args.stage == 2:
        # STAGE 2
        if args.svcnn_name == "":
            print("SVCNN Model name required")
            sys.exit()

        wandb.init(
            project=project_name,
            entity="icheler-team",
            resume=args.continue_training,
            name=f"{args.name}_stage_2",
            tags=["stage2"],
            config={
                "name": args.name,
                "num_models": args.num_models,
                "lr": args.lr,
                "batch_size": args.batchSize,
                "weight_decay": args.weight_decay,
                "no_pretraining": args.no_pretraining,
                "cnn_name": args.cnn_name,
                "num_views": args.num_views,
            },
        )

        cnet = SVCNN(
            args.svcnn_name,
            nclasses=40,
            pretraining=False,
            cnn_name=args.cnn_name
        )

        cnet.load(f"runs/{args.svcnn_name}/stage_1")

        cnet_2 = MVCNN(
            args.name,
            cnet,
            nclasses=40,
            cnn_name=args.cnn_name,
            num_views=args.num_views
        )
        del cnet

        log_dir = f"runs/{args.name}/stage_2"
        create_folder(log_dir, args.continue_training)

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
            train_dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=args.num_workers
        )  # shuffle needs to be false! it's done within the trainer

        val_dataset = MultiviewImgDataset(
            args.val_path,
            scale_aug=False,
            rot_aug=False,
            num_views=args.num_views
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=args.num_workers
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
        trainer.train(args.num_epochs)
        wandb.finish()
