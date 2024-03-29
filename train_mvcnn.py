import argparse
import json
import os
import wandb
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tools.img_dataset import ModelNet40Dataset, SNMVDataset, UnifiedDataset
from tools.trainer import ModelNetTrainer, ModelNetTester
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
    "-num_models", type=int, help="number of models per class", default=0
)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest="no_pretraining", action="store_true")
parser.add_argument(
    "-cnn_name", "--cnn_name", type=str, help="cnn model name",
    required=True
)
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument(
    "-dataset",
    type=str,
    default="model_shaded",
    choices=["model_shaded", "model_original", "shapenet", "unified"]
)
parser.add_argument(
    "-num_workers", type=int, default=4
)
parser.add_argument(
    "-num_epochs", type=int, default=1
)
parser.add_argument("-stage", type=str, required=True, choices=["1", "2", "test"])
parser.add_argument("-svcnn_name", type=str, default="")
parser.add_argument("-svcnn_arc", type=str, default="")
parser.add_argument("-resume_id", type=str, default="")
parser.add_argument("-alterations", type=str, nargs="+", default=[],
                    choices=["nopool", "mean", "prepool", "postpool", "freeze"])
parser.set_defaults(train=False)


def create_folder(log_dir, throw_err=True):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    elif throw_err:  # Dont throw error when continue training
        raise ValueError(
            "ERROR: Please change the name of the run model to avoid duplication"
        )
        # print("WARNING: summary folder already exists!! It will be overwritten!!")
        # shutil.rmtree(log_dir)
        # os.mkdir(log_dir)


if __name__ == "__main__":
    project_name = 'r-resnet-shapenet'

    args = parser.parse_args()

    n_models_train = args.num_models * args.num_views
    if args.dataset.startswith("model"):
        n_classes = 40
    elif args.dataset == "shapenet":
        n_classes = 55
    else:
        n_classes = 76

    if not torch.cuda.is_available():
        print("Not using cuda... exiting")
        sys.exit()

    pretraining = not args.no_pretraining
    log_dir = f"runs/{args.name}"
    create_folder(log_dir, False)
    config_f = open(os.path.join(log_dir, "config.json"), "w")
    json.dump(vars(args), config_f)
    config_f.close()

    if args.stage == "1":
        wandb.init(
            id=args.resume_id,
            project=project_name,
            entity="icheler-team",
            resume=args.resume_id != "",
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

        # STAGE 1
        log_dir = f"runs/{args.name}/stage_1"
        create_folder(log_dir, args.resume_id == "")
        cnet = SVCNN(
            args.name,
            nclasses=n_classes,
            pretraining=pretraining,
            cnn_name=args.cnn_name
        )

        optimizer = optim.Adam(
            cnet.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        if args.dataset.startswith("model"):
            train_dataset = ModelNet40Dataset(
                args.dataset,
                dataset='train',
                num_models=n_models_train,
                num_views=1,
                shuffle=True,
            )
            val_dataset = ModelNet40Dataset(
                args.dataset,
                dataset='val',
                num_models=n_models_train,
                num_views=1,
                shuffle=True,
            )
        elif args.dataset == 'shapenet':
            train_dataset = SNMVDataset(
                dataset='train',
                num_models=n_models_train,
                num_views=1
            )
            val_dataset = SNMVDataset(
                dataset='val',
                num_models=n_models_train,
                num_views=1
            )
        else:
            train_dataset = UnifiedDataset(
                args.dataset,
                dataset='train',
                num_modelnet_models=int(n_models_train / 2),
                num_shapenet_models=n_models_train - int(n_models_train / 2),
                num_views=1,
            )
            val_dataset = UnifiedDataset(
                args.dataset,
                dataset='val',
                num_modelnet_models=int(n_models_train / 2),
                num_shapenet_models=n_models_train - int(n_models_train / 2),
                num_views=1,
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchSize,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=args.num_workers,
        )
        print("num_train_files: " + str(len(train_dataset)))
        print("num_val_files: " + str(len(val_dataset)))
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

    elif args.stage == "2":
        # STAGE 2
        if args.svcnn_name == "" or args.svcnn_arc == "":
            print("SVCNN Model name and architecture required")
            sys.exit()

        wandb.init(
            id=args.resume_id,
            project=project_name,
            entity="icheler-team",
            resume=args.resume_id != "",
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
            nclasses=n_classes,
            pretraining=False,
            cnn_name=args.svcnn_arc
        )

        cnet.load(f"runs/{args.svcnn_name}/stage_1")

        cnet_2 = MVCNN(
            args.name,
            cnet,
            alterations=args.alterations,
            nclasses=n_classes,
            cnn_name=args.cnn_name,
            num_views=args.num_views
        )
        del cnet

        log_dir = f"runs/{args.name}/stage_2"
        create_folder(log_dir, args.resume_id == "")

        optimizer = optim.Adam(
            cnet_2.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )

        if args.dataset.startswith("model"):
            train_dataset = ModelNet40Dataset(
                args.dataset,
                dataset='train',
                num_models=n_models_train,
                num_views=args.num_views,
            )
            val_dataset = ModelNet40Dataset(
                args.dataset,
                dataset='val',
                num_models=n_models_train,
                num_views=args.num_views,
            )
        elif args.dataset == 'shapenet':
            train_dataset = SNMVDataset(
                dataset='train',
                num_models=n_models_train,
                num_views=args.num_views
            )
            val_dataset = SNMVDataset(
                dataset='val',
                num_models=n_models_train,
                num_views=args.num_views
            )
        else:
            train_dataset = UnifiedDataset(
                args.dataset,
                dataset='train',
                num_modelnet_models=int(n_models_train / 2),
                num_shapenet_models=n_models_train - int(n_models_train / 2),
                num_views=args.num_views,
            )
            val_dataset = UnifiedDataset(
                args.dataset,
                dataset='val',
                num_modelnet_models=int(n_models_train / 2),
                num_shapenet_models=n_models_train - int(n_models_train / 2),
                num_views=args.num_views,
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchSize,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=args.num_workers,
        )
        print("num_train_files: " + str(len(train_dataset)))
        print("num_val_files: " + str(len(val_dataset)))
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

    elif args.stage == "test":
        if args.svcnn_arc == "":
            print("SVCNN Model architecture required")
            sys.exit()

        wandb.init(
            project=project_name,
            entity="icheler-team",
            name=f"{args.name}_stage_test",
            tags=["test"],
            config={
                "name": args.name,
                "batch_size": args.batchSize,
                "cnn_name": args.cnn_name,
            },
        )

        cnet = SVCNN(
            "Test SVCNN",
            nclasses=n_classes,
            pretraining=False,
            cnn_name=args.svcnn_arc
        )

        cnet_2 = MVCNN(
            args.name,
            cnet,
            alterations=args.alterations,
            nclasses=n_classes,
            cnn_name=args.cnn_name,
            num_views=args.num_views
        )

        cnet_2.load(f"runs/{args.name}/stage_2")

        del cnet

        if args.dataset.startswith("model"):
            test_dataset = ModelNet40Dataset(
                args.dataset,
                dataset='test',
                num_models=n_models_train,
                num_views=args.num_views,
            )
        elif args.dataset == 'shapenet':
            test_dataset = SNMVDataset(
                dataset='test',
                num_models=n_models_train,
                num_views=args.num_views
            )
        else:
            test_dataset = UnifiedDataset(
                args.dataset,
                dataset='test',
                num_modelnet_models=int(n_models_train / 2),
                num_shapenet_models=n_models_train - int(n_models_train / 2),
                num_views=args.num_views,
            )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=args.num_workers,
        )

        tester = ModelNetTester(
            cnet_2,
            test_loader,
            nn.CrossEntropyLoss(),
            log_dir,
            num_views=args.num_views,
        )

        print("num_test_files: " + str(len(test_dataset)))

        tester.test()
        wandb.finish()
