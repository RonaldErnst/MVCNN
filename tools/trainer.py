import numpy as np
import torch
import wandb
import os

# import torch.nn as nn
from torch.autograd import Variable


class ModelNetTrainer(object):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        model_name,
        log_dir,
        num_views=12,
    ):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views
        self.num_classes = self.model.nclasses

        self.model.cuda()

    def train(self, n_epochs):

        best_acc = 0
        i_acc = 0
        self.model.train()
        epoch = 0

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        complete_path = os.path.join(self.log_dir, "checkpoint.pth")
        complete_path = complete_path.replace(os.sep, '/')

        if wandb.run.resumed:
            self.model.load(self.log_dir)
            checkpoint = torch.load(complete_path)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch'] + 1  # Last finished epoch
            loss = checkpoint['loss']
            n_epochs += epoch

        while epoch < n_epochs:
            # permute data for mvcnn TODO necessary? just do it in Dataset?
            # rand_idx = np.random.permutation(
            #     int(len(self.train_loader.dataset.filepaths) / self.num_views)
            # )
            # filepaths_new = []
            # for i in range(len(rand_idx)):
            #     filepaths_new.extend(
            #         self.train_loader.dataset.filepaths[
            #             rand_idx[i]
            #             * self.num_views: (rand_idx[i] + 1)
            #             * self.num_views
            #         ]
            #     )
            # self.train_loader.dataset.filepaths = filepaths_new

            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):

                if self.model_name == "mvcnn":
                    N, V, C, H, W = data[1].size()
                    in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                else:
                    in_data = Variable(data[1].cuda())

                target = Variable(data[0]).cuda().long()

                self.optimizer.zero_grad()

                out_data = self.model(in_data)

                loss = self.loss_fn(out_data, target)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float() / results.size()[0]

                loss.backward()
                self.optimizer.step()

                log_str = ("epoch %d, step %d: train_loss %.3f; train_acc %.3f;"
                           " %.1f%% done") % (
                    epoch + 1,
                    i + 1,
                    loss,
                    acc,
                    i * 100 / len(self.train_loader)
                )
                wandb.log(
                    {
                        "train": {
                            "epoch": epoch + 1,
                            "step": i + 1,
                            "loss": loss,
                            "acc": acc,
                        }
                    }
                )
                if (i + 1) % 1 == 0:
                    print(log_str)
            i_acc += i

            # evaluation
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    (
                        loss,
                        val_overall_acc,
                        val_mean_class_acc,
                    ) = self.update_validation_accuracy(epoch)

            wandb.log(
                    {
                        "val": {
                            "epoch": epoch + 1,
                            "loss": loss,
                            "overall_acc": val_overall_acc,
                            "mean_class_acc": val_mean_class_acc,
                        }
                    }
                )

            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                self.model.save(self.log_dir, epoch)

            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss
            }, complete_path)
            wandb.save(complete_path)  # saves checkpoint to wandb

            epoch += 1

            # adjust learning rate manually
            if epoch > 0 and (epoch + 1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = param_group["lr"] * 0.5

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0

        # in_data = None
        # out_data = None
        # target = None

        wrong_class = np.ones(self.num_classes)
        samples_class = np.ones(self.num_classes)
        all_loss = 0

        self.model.eval()

        # avgpool = nn.AvgPool1d(1, 1)

        # total_time = 0.0
        # total_print_time = 0.0
        # all_target = []
        # all_pred = []

        for index, data in enumerate(self.val_loader, 0):

            if self.model_name == "mvcnn":
                N, V, C, H, W = data[1].size()
                in_data = Variable(data[1]).view(-1, C, H, W).cuda()
            else:  # 'svcnn'
                in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            loss = self.loss_fn(out_data, target).cpu().data.numpy()
            all_loss += loss
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype("int")[i]] += 1
                samples_class[target.cpu().data.numpy().astype("int")[i]] += 1
            correct_points = torch.sum(results.long())

            curr_acc = correct_points / results.size()[0]
            all_correct_points += correct_points
            all_points += results.size()[0]

            log_str = "epoch %d, step %d: val_loss %.3f; val_acc %.3f; %.1f%% done;" % (
                    epoch + 1,
                    index + 1,
                    loss,
                    curr_acc,
                    index * 100 / len(self.val_loader)
                )

            print(log_str)

            wandb.log(
                    {
                        "val": {
                            "epoch": epoch + 1,
                            "step": index + 1,
                            "loss": loss,
                            "acc": curr_acc,
                        }
                    }
                )

        print("Total # of test models: ", all_points)
        val_mean_class_acc = np.mean((samples_class - wrong_class) / samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print("val mean class acc. : ", val_mean_class_acc)
        print("val overall acc. : ", val_overall_acc)
        print("val loss : ", loss)

        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc
