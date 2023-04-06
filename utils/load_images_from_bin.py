#!/usr/bin/env python
# encoding: utf-8



import os
import argparse
import pickle
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import mxnet as mx


def load_mx_rec(rec_path):
    save_path = os.path.join(rec_path, 'emore_images')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    imgrec = mx.recordio.MXIndexedRecordIO(os.path.join(rec_path, 'train.idx'), os.path.join(rec_path, 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #img = Image.fromarray(img)
        label_path = os.path.join(save_path, str(label).zfill(6))
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        #img.save(os.path.join(label_path, str(idx).zfill(8) + '.jpg'), quality=95)
        cv2.imwrite(os.path.join(label_path, str(idx).zfill(8) + '.jpg'), img)


def load_image_from_bin(bin_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file = open(os.path.join(save_dir, '../', 'cfp_fp_pair.txt'), 'w')
    bins, issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')
    for idx in tqdm(range(len(bins))):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, str(idx+1).zfill(5)+'.jpg'), img)
        if idx % 2 == 0:
            label = 1 if issame_list[idx//2] else -1
            file.write(str(idx+1).zfill(5) + '.jpg' + ' ' + str(idx+2).zfill(5) +'.jpg' + ' ' + str(label) + '\n')


def visualize_dataset(src_dir):

    data = {}
    for id_dir in os.listdir(src_dir):
        data[id_dir] = len(os.listdir(os.path.join(src_dir, id_dir)))

    ids = list(data.keys())
    values = list(data.values())

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(ids, values, color='maroon', width=0.4)

    plt.xlabel("Identities")
    plt.ylabel("Images per identity")
    plt.title("Face Recognition Dataset")
    # plt.show()
    plt.savefig('showcase.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load image from Binary based dataset')
    parser.add_argument('--bin_path', type=str, default="/home/Dev-1/datasets/faces_emore/cfp_fp.bin")
    parser.add_argument('--save_dir', type=str, default="/home/Dev-1/datasets/faces_emore/cfp_fp")
    parser.add_argument('--rec_path', type=str, default="D:/face_data_emore/faces_emore")
    args = parser.parse_args()

    # load_mx_rec(args)
    load_image_from_bin(args.bin_path, args.save_dir)
    # visualize_dataset(args.save_dir)
