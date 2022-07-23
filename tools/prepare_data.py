import numpy as np
import pandas as pd
import os
import shutil
import sys

from tarfile import open


root_dir = '../data'
shapenet_tar = os.path.join(root_dir, 'shapenet55v1.tar')
shapenet_dir = os.path.join(root_dir, 'shapenet55v1')

def start(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

def done():
    print('Done')

assert not os.path.exists(shapenet_dir)

start('Extracting from ' + shapenet_tar + '...')
with open(shapenet_tar) as tar:
    tar.extractall(root_dir)
done()

start('Reading and splitting csv...')
def snp(p):
    return os.path.join(shapenet_dir, p)

train = pd.read_csv(snp('train.csv'))
train = train[train['id'] != 4004]
val = pd.read_csv(snp('val.csv'))
cat = pd.concat([train, val])

source_dirs = { id : snp('train') for id in train['id'] }
source_dirs.update({ id: snp('val') for id in val['id'] })

cat = cat.sample(frac=1, random_state=10538)
train, val, test = np.split(cat, [int(0.6 * len(cat)), int(0.8 * len(cat))])

train.to_csv(snp('train.csv'))
val.to_csv(snp('val.csv'))
test.to_csv(snp('test.csv'))
done()

def images(id):
    for i in range(1, 13):
        yield f'model_{id:06d}_{i:03d}.jpg'

start('Moving images...')
def move_to_dir(ids, dst_dir):
    for id in ids:
        src_dir = source_dirs[id]
        for file_name in images(id):
            src_path = os.path.join(src_dir, file_name)
            dst_path = os.path.join(dst_dir, file_name)
            shutil.move(src_path, dst_path)

move_to_dir(train['id'], snp('train'))
move_to_dir(val['id'], snp('val'))
shutil.rmtree(snp('test'))
os.mkdir(snp('test'))
move_to_dir(test['id'], snp('test'))
done()

start('Deleting unlabeled images...')
def filter_images_by_ids(dir_path, ids):
    for jpg in os.listdir(dir_path):
        id = int(jpg.split('_')[1])
        if id not in ids:
            os.remove(os.path.join(dir_path, jpg))

filter_images_by_ids(snp('train'), set(train['id']))
filter_images_by_ids(snp('val'), set(val['id']))
filter_images_by_ids(snp('test'), set(test['id']))
done()
