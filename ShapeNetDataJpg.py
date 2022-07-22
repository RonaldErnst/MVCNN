import numpy as np
import pandas as pd

from os import path
from PIL import Image
from torch.utils.data import Dataset
from json import loads

class SNMVDataset(Dataset):
    def __init__(self, dir_path, dataset):
        assert dataset in ('train', 'val')

        with open(path.join(dir_path, dataset + '.csv')) as csv:
            self.csv = pd.read_csv(csv)
            self.csv = self.csv[self.csv['id'] != 4004].reset_index(drop=True)
            self.csv['id'] = self.csv['id'].astype('int')
        self.dir_path = path.join(dir_path, dataset)
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        item = self.csv[self.csv.index == idx]
        model_path_base = f'model_{int(item["id"].item()):06d}_'
        model_path_base = path.join(self.dir_path, model_path_base)
        img_array = np.zeros((12, 224, 224, 3))
        for i in range(12):
            model_path = model_path_base + f'{i + 1:03d}.jpg'
            with open(model_path, 'rb') as jpg:
                jpg = Image.open(jpg)
                img_array[i] = np.asarray(jpg)
        return img_array, self._get_label(item['synsetId'].item(),
                                          item['subSynsetId'].item())
    
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


def iterate_jpg(dataset):
    for i in range(len(dataset)):
        dataset[i]

def iterate_dataset(dataset):
    iterate_jpg(SNMVDataset('ShapeNet/shapenet55v1', dataset))
    iterate_jpg(SNMVNameDataset('ShapeNet/shapenet55v1', 'ShapeNet/taxonomy.json', dataset))
    iterate_jpg(SNMVClassDataset('ShapeNet/shapenet55v1', dataset))

iterate_dataset('val')
iterate_dataset('train')
