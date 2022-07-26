from tkinter import W
import numpy as np
import pandas as pd
import os
import shutil
import sys
from tarfile import open

root_dir = '../data'
shapenet_tar = os.path.join(root_dir, 'shapenet55v1.tar')
shapenet_dir = os.path.join(root_dir, 'shapenet55v1')
assert not os.path.exists(shapenet_dir)


def start(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()


def done():
    print('Done')


def snp(p):
    return os.path.join(shapenet_dir, p)


def images(id):
    for i in range(1, 13):
        yield f'model_{id:06d}_{i:03d}.jpg'


def move_to_dir(source_dirs, ids, dst_dir):
    for id in ids:
        src_dir = source_dirs[id]
        for file_name in images(id):
            src_path = os.path.join(src_dir, file_name)
            dst_path = os.path.join(dst_dir, file_name)
            shutil.move(src_path, dst_path)


def filter_images_by_ids(dir_path, ids):
    for jpg in os.listdir(dir_path):
        id = int(jpg.split('_')[1])
        if id not in ids:
            os.remove(os.path.join(dir_path, jpg))


if __name__ == "__main__":
    start('Extracting from ' + shapenet_tar + '...')
    with open(shapenet_tar) as tar:
        tar.extractall(root_dir)
    done()

    start('Reading and splitting csv...')

    train = pd.read_csv(snp('train.csv'))
    train = train[train['id'] != 4004]
    val = pd.read_csv(snp('val.csv'))
    cat = pd.concat([train, val])

    source_dirs = {id: snp('train') for id in train['id']}
    source_dirs.update({id: snp('val') for id in val['id']})

    train = []
    val = []
    test = []
    for cls in cat['synsetId'].unique():
        cls_df = cat[cat['synsetId'] == cls]
        cls_train, cls_val, cls_test = np.split(cls_df, [int(0.6 * len(cls_df)), int(0.8 * len(cls_df))])
        train += cls_train.values.tolist()
        val += cls_val.values.tolist()
        test += cls_test.values.tolist()
    train = pd.DataFrame(train, columns=cat.columns)
    val = pd.DataFrame(val, columns=cat.columns)
    test = pd.DataFrame(test, columns=cat.columns)

    train.to_csv(snp('train.csv'))
    val.to_csv(snp('val.csv'))
    test.to_csv(snp('test.csv'))
    done()

    start('Moving images...')
    move_to_dir(source_dirs, train['id'], snp('train'))
    move_to_dir(source_dirs, val['id'], snp('val'))
    shutil.rmtree(snp('test'))
    os.mkdir(snp('test'))
    move_to_dir(source_dirs, test['id'], snp('test'))
    done()

    start('Deleting unlabeled images...')
    filter_images_by_ids(snp('train'), set(train['id']))
    filter_images_by_ids(snp('val'), set(val['id']))
    filter_images_by_ids(snp('test'), set(test['id']))
    done()
