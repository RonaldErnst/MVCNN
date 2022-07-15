import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from .model import Model

mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :,
        getattr(
            torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda]
        )().long(),
        :,
    ]
    return x.view(xsize)


class SVCNN(Model):
    def __init__(self, name, nclasses=40, pretraining=True, cnn_name="resnet18"):
        super(SVCNN, self).__init__(name)

        self.classnames = [
            "airplane",
            "bathtub",
            "bed",
            "bench",
            "bookshelf",
            "bottle",
            "bowl",
            "car",
            "chair",
            "cone",
            "cup",
            "curtain",
            "desk",
            "door",
            "dresser",
            "flower_pot",
            "glass_box",
            "guitar",
            "keyboard",
            "lamp",
            "laptop",
            "mantel",
            "monitor",
            "night_stand",
            "person",
            "piano",
            "plant",
            "radio",
            "range_hood",
            "sink",
            "sofa",
            "stairs",
            "stool",
            "table",
            "tent",
            "toilet",
            "tv_stand",
            "vase",
            "wardrobe",
            "xbox",
        ]

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith("resnet")
        self.mean = Variable(
            torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False
        ).cuda()
        self.std = Variable(
            torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False
        ).cuda()

        if self.use_resnet:
            if self.cnn_name == "resnet18":
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == "resnet34":
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == "resnet50":
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, self.nclasses)
        else:
            if self.cnn_name == "inception":
                self.net = models.inception_v3(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, self.nclasses)
            elif self.cnn_name == "convnext":
                self.net = models.convnext_base(pretrained=self.pretraining)
                self.net.classifier._modules["2"] = nn.Linear(1024, self.nclasses)

    def forward(self, x):
        return self.net(x)


class MVCNN(Model):
    def __init__(self, name, model, nclasses=40, cnn_name="resnet18", num_views=12):
        super(MVCNN, self).__init__(name)

        self.classnames = [
            "airplane",
            "bathtub",
            "bed",
            "bench",
            "bookshelf",
            "bottle",
            "bowl",
            "car",
            "chair",
            "cone",
            "cup",
            "curtain",
            "desk",
            "door",
            "dresser",
            "flower_pot",
            "glass_box",
            "guitar",
            "keyboard",
            "lamp",
            "laptop",
            "mantel",
            "monitor",
            "night_stand",
            "person",
            "piano",
            "plant",
            "radio",
            "range_hood",
            "sink",
            "sofa",
            "stairs",
            "stool",
            "table",
            "tent",
            "toilet",
            "tv_stand",
            "vase",
            "wardrobe",
            "xbox",
        ]

        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(
            torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False
        ).cuda()
        self.std = Variable(
            torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False
        ).cuda()

        self.use_resnet = cnn_name.startswith("resnet")

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            if self.cnn_name == "inception":
                self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
                self.net_2 = model.net.fc
            elif self.cnn_name == "convnext":
                self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
                self.net_2 = model.net.classifier._modules["2"]

    def forward(self, x):
        y = self.net_1(x)
        y = y.view(
            (
                int(x.shape[0] / self.num_views),
                self.num_views,
                y.shape[-3],
                y.shape[-2],
                y.shape[-1],
            )
        )  # (8,12,512,7,7)
        return self.net_2(torch.max(y, 1)[0].view(y.shape[0], -1))
