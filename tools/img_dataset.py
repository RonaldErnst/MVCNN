import glob
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import csv


class MultiviewImgDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        scale_aug=False,
        rot_aug=False,
        test_mode=False,
        num_models=0,
        num_views=12,
        shuffle=True,
    ):
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
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views

        set_ = root_dir.split("/")[-1]
        parent_dir = root_dir.rsplit("/", 2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(
                glob.glob(parent_dir + "/" + self.classnames[i] + "/" + set_ + "/*.png")
            )
            # Select subset for different number of views
            stride = int(12 / self.num_views)  # 12 6 4 3 2 1
            all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[: min(num_models, len(all_files))])

        if shuffle:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths) / num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(
                    self.filepaths[
                        rand_idx[i] * num_views: (rand_idx[i] + 1) * num_views
                    ]
                )
            self.filepaths = filepaths_new

        if self.test_mode:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return int(len(self.filepaths) / self.num_views)

    def __getitem__(self, idx):
        path = self.filepaths[idx * self.num_views]
        path = path.replace(os.sep, '/')
        class_name = path.split("/")[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx * self.num_views + i]).convert("RGB")
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (
            class_id,
            torch.stack(imgs),
            self.filepaths[idx * self.num_views: (idx + 1) * self.num_views],
        )


class SingleImgDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        scale_aug=False,
        rot_aug=False,
        test_mode=False,
        num_models=0,
        num_views=12,
    ):
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
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode

        set_ = root_dir.split("/")[-1]
        parent_dir = root_dir.rsplit("/", 2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(
                glob.glob(
                    parent_dir + "/" + self.classnames[i] + "/" + set_ + "/*shaded*.png"
                )
            )
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[: min(num_models, len(all_files))])

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split("/")[-3]
        class_id = self.classnames.index(class_name)

        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert("RGB")
        if self.transform:
            im = self.transform(im)

        return (class_id, im, path)


class ModelNet40Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        version,
        train=True,
        num_models=0,
        num_views=1,  # num_views = 1 => Singleview; num_views > 1 => Multiview
        shuffle=False
    ):
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
        self.num_classes = len(self.classnames)
        self.num_views = num_views
        self.is_multiview = self.num_views > 1

        if version == "model_shaded":
            if train:
                self.root_dir = "data/modelnet40_images_new_12x/*/train"
            else:
                self.root_dir = "data/modelnet40_images_new_12x/*/test"
        else:
            if train:
                self.root_dir = "data/modelnet40_original_12x/*/train"
            else:
                self.root_dir = "data/modelnet40_original_12x/*/test"

        fileformat = "*shaded*.png" if version == "model_shaded" else "*.jpg"

        set_ = self.root_dir.split("/")[-1]
        parent_dir = self.root_dir.rsplit("/", 2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(
                glob.glob(
                    parent_dir + "/" + self.classnames[i] + "/" +
                    set_ + "/" + fileformat
                )
            )

            if self.is_multiview:
                # Select subset for different number of views
                stride = int(12 / self.num_views)  # 12 6 4 3 2 1
                all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[: min(num_models, len(all_files))])

        if shuffle:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths) / self.num_views))
            filepaths_new = []
            for i in rand_idx:
                filepaths_new.extend(
                    self.filepaths[i * self.num_views: (i + 1) * self.num_views]
                )
            self.filepaths = filepaths_new

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return int(len(self.filepaths) / self.num_views)

    def __getitem__(self, idx):
        path = self.filepaths[idx * self.num_views]
        path = path.replace(os.sep, '/')
        class_name = path.split("/")[-3]
        class_id = self.classnames.index(class_name)

        if self.is_multiview:
            # Use PIL instead
            imgs = []
            for i in range(self.num_views):
                im = Image.open(self.filepaths[idx * self.num_views + i]).convert("RGB")
                if self.transform:
                    im = self.transform(im)
                imgs.append(im)

            return (
                class_id,
                torch.stack(imgs),
                [p.replace(os.sep, '/') for p in
                    self.filepaths[idx * self.num_views:(idx + 1) * self.num_views]],
            )
        else:
            # Use PIL instead
            im = Image.open(self.filepaths[idx]).convert("RGB")
            if self.transform:
                im = self.transform(im)

            return (class_id, im, path)


class ShapeNet55Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train=True,
        num_models=0,
        num_views=1,  # num_views = 1 => Singleview; num_views > 1 => Multiview
        shuffle=False
    ):
        self.classnames = [
            '02691156',
            '02747177',
            '02773838',
            '02801938',
            '02808440',
            '02818832',
            '02828884',
            '02843684',
            '02871439',
            '02876657',
            '02880940',
            '02924116',
            '02933112',
            '02942699',
            '02946921',
            '02954340',
            '02958343',
            '02992529',
            '03001627',
            '03046257',
            '03085013',
            '03207941',
            '03211117',
            '03261776',
            '03325088',
            '03337140',
            '03467517',
            '03513137',
            '03593526',
            '03624134',
            '03636649',
            '03642806',
            '03691459',
            '03710193',
            '03759954',
            '03761084',
            '03790512',
            '03797390',
            '03928116',
            '03938244',
            '03948459',
            '03991062',
            '04004475',
            '04074963',
            '04090263',
            '04099429',
            '04225987',
            '04256520',
            '04330267',
            '04379243',
            '04401088',
            '04460130',
            '04468005',
            '04530566',
            '04554684',
        ]
        self.num_views = num_views
        self.is_multiview = self.num_views > 1
        self.num_classes = len(self.classnames)

        if train:
            self.root_dir = "data/shapenet55v1/train"
            self.classmap_dir = "data/shapenet55v1/train.csv"
        else:
            self.root_dir = "data/shapenet55v1/val"
            self.classmap_dir = "data/shapenet55v1/val.csv"

        self.classmap = self.__load_csv__(self.classmap_dir)

        set_ = self.root_dir.split("/")[-1]
        parent_dir = self.root_dir.rsplit("/", 1)[0]
        self.filepaths = []

        all_files = sorted(
            glob.glob(parent_dir + "/" + set_ + "/*.jpg")
        )

        # Filter out all that are not listed in classmap
        invalid_files = []
        for path in all_files:
            filename = path.split("/")[-1]
            obj_id = filename.split("_")[1]

            if obj_id not in self.classmap:
                invalid_files.append(path)

        print("Filtering out bad files...")
        all_files = [p for p in all_files if p not in invalid_files]
        print("Done")

        if self.is_multiview:
            # Select subset for different number of views
            stride = int(12 / self.num_views)  # 12 6 4 3 2 1
            all_files = all_files[::stride]

        if num_models == 0:
            # Use the whole dataset
            self.filepaths.extend(all_files)
        else:
            self.filepaths.extend(all_files[: min(num_models, len(all_files))])

        if shuffle:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths) / num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(
                    self.filepaths[
                        rand_idx[i] * num_views: (rand_idx[i] + 1) * num_views
                    ]
                )
            self.filepaths = filepaths_new

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __load_csv__(self, dir):
        classmap = {}

        with open(dir) as file:
            csvfile = csv.reader(file)
            for row in csvfile:
                obj_id, synsetId, _ = row

                if obj_id in classmap:
                    raise Exception("Duplicate Object ID ", obj_id)

                classmap[obj_id] = synsetId

        return classmap

    def __len__(self):
        return int(len(self.filepaths) / self.num_views)

    def __getitem__(self, idx):
        path = self.filepaths[idx * self.num_views]
        path = path.replace(os.sep, '/')

        filename = path.split("/")[-1]
        obj_id = filename.split("_")[1]
        class_name = self.classmap[obj_id]
        class_id = self.classnames.index(class_name)

        if self.is_multiview:
            # Use PIL instead
            imgs = []
            for i in range(self.num_views):
                im = Image.open(self.filepaths[idx * self.num_views + i]).convert("RGB")
                if self.transform:
                    im = self.transform(im)
                imgs.append(im)

            return (
                class_id,
                torch.stack(imgs),
                [p.replace(os.sep, '/') for p in
                    self.filepaths[idx * self.num_views:(idx + 1) * self.num_views]],
            )
        else:
            # Use PIL instead
            im = Image.open(self.filepaths[idx]).convert("RGB")
            if self.transform:
                im = self.transform(im)

            return (class_id, im, path)
