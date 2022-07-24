import os

import numpy as np

from shutil import move
from sys import stdout
from tarfile import open


root_dir = '../data'
modelnet_tar = os.path.join(root_dir, 'modelnet40v1.tar')
modelnet_dir = os.path.join(root_dir, 'modelnet40_original_12x')
modelnet_dir_orig = os.path.join(root_dir, 'modelnet40v1')

def start(msg):
    stdout.write(msg)
    stdout.flush()

def done():
    print('Done')

assert not os.path.exists(modelnet_dir)

start('Extracting from ' + modelnet_tar + '...')
with open(modelnet_tar) as tar:
    tar.extractall(root_dir)
move(modelnet_dir_orig, modelnet_dir)
done()

def mnp(p):
    return os.path.join(modelnet_dir, p)

start('Re-splitting per class:')
classes = os.listdir(modelnet_dir)
for i, cls in enumerate(classes):
    start(f'{i}/{len(classes)} - {cls}\n')
    cls_dir = os.path.join(modelnet_dir, cls)

    def collect_ids(dir):
        dir = os.path.join(cls_dir, dir)
        return set('_'.join(jpg.split('_')[:-1]) for jpg in os.listdir(dir))

    src_test_ids = collect_ids('test')
    src_train_ids = collect_ids('train')

    src_dir = {id : 'test' for id in src_test_ids}
    src_dir.update({id : 'train' for id in src_train_ids})

    all_ids = sorted(list(set.union(src_test_ids, src_train_ids)))
    all_ids = np.array(all_ids)
    np.random.RandomState(10538).shuffle(all_ids)
    dst_train_ids, dst_val_ids, dst_test_ids = \
        np.split(all_ids, [int(0.6 * len(all_ids)), int(0.8 * len(all_ids))])

    def jpg_names(id):
        return [f'{id}_{i:03}.jpg' for i in range(1, 13)]

    def move_to_dir(ids, dst):
        dst = os.path.join(modelnet_dir, cls, dst)
        for id in ids:
            for jpg in jpg_names(id):
                src = os.path.join(modelnet_dir, cls, src_dir[id], jpg)
                move(src, os.path.join(dst, jpg))
    
    move_to_dir(dst_train_ids, 'train')
    os.mkdir(os.path.join(modelnet_dir, cls, 'val'))
    move_to_dir(dst_val_ids, 'val')
    move_to_dir(dst_test_ids, 'test')

done()
