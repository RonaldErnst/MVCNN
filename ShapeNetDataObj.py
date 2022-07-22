import os
from torch.utils.data import Dataset
from zipfile import ZipFile
from pathlib import Path
from json import loads
from os.path import join

from render import ObjMultiViewRenderer

class SNMVDataset(Dataset):
    def __init__(self, archive_path, device, image_size=256):
        self.archive = ZipFile(archive_path, 'r')
        self.renderer = ObjMultiViewRenderer(self.archive, device, image_size)

        self.model_paths = [fn for fn in self.archive.namelist() if fn.endswith('obj')]
    
    def _load_taxonomy(self):
        taxonomy = self.archive.read(join('ShapeNetCore.v2', 'taxonomy.json'))
        return loads(taxonomy)

    def __len__(self):
        return len(self.model_paths)
    
    def __getitem__(self, idx):
        model_path = self.model_paths[idx]
        dirname = os.path.dirname(model_path)
        images = self.renderer.render(model_path, dirname)
        label = self._get_label(model_path)
        return images, label

    def __enter__(self):
        return self

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     self.archive.close()
    
    def _get_label(self, model_path):
        return model_path.split('/')[1]


class SNMVNameDataset(SNMVDataset):
    def __init__(self, archive_path, device, image_size=256):
        super().__init__(archive_path, device, image_size)
        self.label_dict = {cls['synsetId']: cls['name'] for cls in self._load_taxonomy()}

    def _get_label(self, model_path):
        synsetId = super()._get_label(model_path)
        return self.label_dict[synsetId]


class SNMVClassDataset(SNMVDataset):
    def __init__(self, archive_path, device, image_size=256):
        super().__init__(archive_path, device, image_size)
        synsetIds = set(super(SNMVClassDataset, self)._get_label(path) for path in self.model_paths)
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
        self.model_paths = [path for path in self.model_paths \
            if super(SNMVClassDataset, self)._get_label(path) in self.label_dict]

    def _get_label(self, model_path):
        synsetId = super()._get_label(model_path)
        return self.label_dict[synsetId]
