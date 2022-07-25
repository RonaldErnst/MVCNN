import os
import numpy as np
from shutil import move
from sys import stdout
from tarfile import open


root_dir = '../data'
modelnets = [('modelnet40v1.tar', 'modelnet40_original_12x'),
             ("shaded_images.tar.gz", 'modelnet40_images_new_12x')]


def start(msg):
    stdout.write(msg)
    stdout.flush()


def done():
    print('Done')


def collect_ids(cls_dir, dir):
    dir = os.path.join(cls_dir, dir).replace(os.sep, "/")

    # Delete all files that start with a dot. Usually they cannot be used
    for jpg in os.listdir(dir):
        if not os.path.isdir(dir + "/" + jpg) and jpg.startswith("."):
            os.remove(dir + "/" + jpg)

    return set('_'.join(jpg.split('_')[:-1]) for jpg in os.listdir(dir))


def jpg_names(id):
    return [f'{id}_{i:03}.jpg' for i in range(1, 13)]


def png_names(id):
    return [f'{id}_v{i:03}.png' for i in range(1, 13)]


def move_to_dir(modelnet_dir, cls, src_dir, ids, dst):
    dst = os.path.join(modelnet_dir, cls, dst).replace(os.sep, "/")
    for id in ids:
        if modelnet_dir.endswith("original_12x"):
            images = jpg_names(id)
        else:
            images = png_names(id)

        for img in images:
            src = os.path.join(modelnet_dir, cls, src_dir[id], img).replace(os.sep, "/")
            move(src, os.path.join(dst, img).replace(os.sep, "/"))


if __name__ == "__main__":
    for modelnet in modelnets:
        modelnet_tar, modelnet_dir = modelnet
        modelnet_dir_orig = root_dir + "/" + modelnet_tar.rsplit(".", 2)[0]

        modelnet_dir = os.path.join(root_dir, modelnet_dir).replace(os.sep, "/")
        modelnet_tar = os.path.join(root_dir, modelnet_tar).replace(os.sep, "/")
        assert not os.path.exists(modelnet_dir)

        start('Extracting from ' + modelnet_tar + '...')
        with open(modelnet_tar) as tar:
            tar.extractall(root_dir)

        if modelnet_tar.endswith("modelnet40v1.tar"):
            move(modelnet_dir_orig, modelnet_dir)
        done()

        start('Re-splitting per class:')
        classes = [f for f in os.listdir(modelnet_dir)
                   if os.path.isdir(modelnet_dir + "/" + f)]
        for i, cls in enumerate(classes):
            start(f'{i}/{len(classes)} - {cls}\n')
            cls_dir = os.path.join(modelnet_dir, cls).replace(os.sep, "/")

            src_test_ids = collect_ids(cls_dir, 'test')
            src_train_ids = collect_ids(cls_dir, 'train')

            src_dir = {id: 'test' for id in src_test_ids}
            src_dir.update({id: 'train' for id in src_train_ids})

            all_ids = sorted(list(set.union(src_test_ids, src_train_ids)))
            all_ids = np.array(all_ids)
            np.random.RandomState(10538).shuffle(all_ids)
            dst_train_ids, dst_val_ids, dst_test_ids = \
                np.split(all_ids, [int(0.6 * len(all_ids)), int(0.8 * len(all_ids))])

            move_to_dir(modelnet_dir, cls, src_dir, dst_train_ids, 'train')
            os.mkdir(os.path.join(modelnet_dir, cls, 'val'))
            move_to_dir(modelnet_dir, cls, src_dir, dst_val_ids, 'val')
            move_to_dir(modelnet_dir, cls, src_dir, dst_test_ids, 'test')

        done()
