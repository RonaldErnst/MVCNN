from re import sub
import numpy as np
import pandas as pd

from os import path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from json import loads

class SNMVDataset(Dataset):
    def __init__(self, dir_path, dataset, num_views):
        assert dataset in ('train', 'val', 'test')

        with open(path.join(dir_path, dataset + '.csv')) as csv:
            csv = pd.read_csv(csv)
        self.filepaths = []
        for item in csv.itertuples(index=False):
            _, id, synsetId, subSynsetId = item
            jpg_path_base = path.join(dir_path, dataset, f'model_{id:06d}_')
            jpg_paths = []
            for i in range(1, 13, int(12 / num_views)):
                jpg_paths.append(jpg_path_base + f'{i:03}.jpg')
            self.filepaths.append((jpg_paths, synsetId, subSynsetId))
        self.dir_path = path.join(dir_path, dataset)
        self.num_views = num_views

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
        jpg_paths, synsetId, subSynsetId = self.filepaths[int(idx * self.num_views)]
        img_array = np.zeros((self.num_views, 3, 224, 224), dtype=np.float32)
        for i, path in enumerate(jpg_paths):
            jpg = Image.open(path)
            jpg = self.transform(jpg)
            img_array[i] = np.asarray(jpg)
        return self._get_label(synsetId, subSynsetId), img_array.squeeze(), jpg_paths
    
    def _get_label(self, synsetId, _):
        return synsetId


class SNMVNameDataset(SNMVDataset):
    def __init__(self, dir_path, taxonomy_path, dataset):
        super().__init__(dir_path, dataset)
        with open(path.join(taxonomy_path)) as taxonomy:
            taxonomy = taxonomy.read()
        taxonomy = loads(taxonomy)
        self.taxonomy = { int(i['synsetId']): i['name'] for i in taxonomy }
    
    def _get_label(self, synsetId, subSynsetId):
        return (self.taxonomy[synsetId], self.taxonomy[subSynsetId])

class SNMVClassDataset(SNMVDataset):
    def __init__(self, dir_path, dataset):
        super().__init__(dir_path, dataset)
        self.label_dict = {
            '02691156': 'airplane', #'airplane,aeroplane,plane',
            # '02747177': 'ashcan,trash can,garbage can,wastebin,ash '
            #             'bin,ash-bin,ashbin,dustbin,trash barrel,trash bin',
            # '02773838': 'bag,traveling bag,travelling bag,grip,suitcase',
            # '02801938': 'basket,handbasket',
            '02808440': 'bathtub', #'bathtub,bathing tub,bath,tub',
            '02818832': 'bed', # 'bed'
            '02828884': 'bench', # 'bench'
            '02843684': 'birdhouse',
            '02871439': 'bookshelf', # 'bookshelf'
            '02876657': 'bottle', # 'bottle'
            '02880940': 'bowl', # 'bowl'
            # '02924116': 'bus,autobus,coach,charabanc,double-decker,jitney,motorbus,motorcoach,omnibus,passenger '
            #             'vehi',
            # '02933112': 'cabinet',
            # '02942699': 'camera,photographic camera',
            # '02946921': 'can,tin,tin can',
            # '02954340': 'cap',
            '02958343': 'car', #'car,auto,automobile,machine,motorcar',
            # '02992529': 'cellular telephone,cellular phone,cellphone,cell,mobile phone',
            '03001627': 'chair', #'chair',
            # '03046257': 'clock',
            '03085013': 'keyboard', #'computer keyboard,keypad',
            # '03207941': 'dishwasher,dish washer,dishwashing machine',
            '03211117': 'monitor', #'display,video display',
            # '03261776': 'earphone,earpiece,headphone,phone',
            # '03325088': 'faucet,spigot',
            # '03337140': 'file,file cabinet,filing cabinet',
            '03467517': 'guitar', #'guitar',
            # '03513137': 'helmet',
            # '03593526': 'jar',
            # '03624134': 'knife',
            '03636649': 'lamp', #'lamp',
            '03642806': 'laptop', #'laptop,laptop computer',
            # '03691459': 'loudspeaker,speaker,speaker unit,loudspeaker system,speaker '
            #             'system',
            # '03710193': 'mailbox,letter box',
            # '03759954': 'microphone,mike',
            # '03761084': 'microwave,microwave oven',
            # '03790512': 'motorcycle,bike',
            '03797390': 'cup', #'mug',
            '03928116': 'piano', #'piano,pianoforte,forte-piano',
            # '03938244': 'pillow',
            # '03948459': 'pistol,handgun,side arm,shooting iron',
            '03991062': 'flower_pot', #'pot,flowerpot',
            # '04004475': 'printer,printing machine',
            # '04074963': 'remote control,remote',
            # '04090263': 'rifle',
            # '04099429': 'rocket,projectile',
            # '04225987': 'skateboard',
            '04256520': 'sofa', #'sofa,couch,lounge',
            # '04330267': 'stove',
            '04379243': 'table', #'table',
            # '04401088': 'telephone,phone,telephone set',
            # '04460130': 'tower',
            # '04468005': 'train,railroad train',
            # '04530566': 'vessel,watercraft',
            # '04554684': 'washer,automatic washer,washing machine'
        }
        self.label_dict = { int(id): name 
            for id, name in self.label_dict.items() }
        self.csv = self.csv[
            self.csv['synsetId'].isin(pd.Series(self.label_dict.keys()))].reset_index(drop=True)

    def _get_label(self, synsetId, _):
        return self.label_dict[synsetId]
