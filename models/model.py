import glob
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name

    def save(self, path, epoch=0):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(
            self.state_dict(),
            os.path.join(path, "model-{}.pth".format(str(epoch).zfill(5))),
        )

    def save_results(self, path, data):
        raise NotImplementedError("Model subclass must implement this method.")

    def load(self, path, modelfile=None):
        if not os.path.exists(path):
            raise IOError("{} directory does not exist in {}".format(self.name, path))

        if modelfile is None:
            model_files = glob.glob(path + "/*")
            mf = max(model_files)
        else:
            mf = os.path.join(path, modelfile)

        self.load_state_dict(torch.load(mf))
