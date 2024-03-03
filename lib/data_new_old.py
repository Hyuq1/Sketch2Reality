#!/usr/bin/env python3

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import pdb
import imageio
import time
import random
import imageio

from lib.utils import *




def azimuth_convert(azimuth):
    if azimuth < 0:
        azimuth = - azimuth
    else:
        azimuth = 360 - azimuth
    return azimuth

def get_camera_matrices1(metadata_filename, id):
    # Adaptation of Code/Utils from DISN
    with open(metadata_filename, 'r') as f:
        lines = f.read().splitlines()
        # param_lst = read_params(lines)
        param_lst = lines[0].split()
        rot_mat = get_rotate_matrix(-np.pi / 2)
        az, el, distance_ratio = azimuth_convert(int(param_lst[0])), int(param_lst[1]), float(param_lst[3]) / 1.75
        # az, el, distance_ratio = azimuth_convert()
        intrinsic, RT = getBlenderProj(az, el, distance_ratio, img_w=224, img_h=224)
        extrinsic = np.linalg.multi_dot([RT, rot_mat])
    intrinsic = torch.tensor(intrinsic).float()
    extrinsic = torch.tensor(extrinsic).float()

    return intrinsic, extrinsic


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample
    ):
        self.subsample = subsample
        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)


    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        return unpack_sdf_samples(filename, self.subsample), idx, self.npyfiles[idx]


class RGBA2SDF(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        # self.npyfiles =  get_instance_filenames(data_source, split)
        self.data_path = os.path.join(self.data_source, '02691156')
        self.sketch_path_source = '/public2/home/huyuanqi/data/shapenet-sketch'
        #self.folder_names = sorted(os.listdir(self.data_path))


    def __len__(self):
        return len(self.folder_names)

    def __getitem__(self, idx):

        # mesh_name = self.npyfiles[idx].split(".npz")[0]
        mesh_name = self.folder_names[idx]

        # fetch sdf samples
        sdf_filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        #sdf_filename = '/public2/home/huyuanqi/data/MeshSDF/airplane_new/samples/1a04e3eab45ca15dd86060f189eb133.npz'
        sdf_samples = unpack_sdf_samples(sdf_filename,  self.subsample)

        if self.is_train:
            # reset seed for random sampling training data (see https://github.com/pytorch/pytorch/issues/5059)
            np.random.seed( int(time.time()) + idx)
            id = np.random.randint(0, self.num_views)
        else:
            np.random.seed(idx)
            id = np.random.randint(0, self.num_views)

        view_id = '{0:02d}'.format(id)


        image_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"),"sketches", view_id + ".png")
        #image_filename = os.path.join(self.data_source, '02691156', mesh_name, 'sketch.png')
        RGBA = unpack_images(image_filename)

        # fetch cameras
        metadata_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), "rendering_metadata.txt")
        #metadata_filename = os.path.join(self.data_source, '02691156', mesh_name, 'view.txt')
        intrinsic, extrinsic = get_camera_matrices1(metadata_filename, id)



        return sdf_samples, RGBA, intrinsic, extrinsic, mesh_name, id
