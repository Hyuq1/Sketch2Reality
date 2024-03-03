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


def azimuth_convert_flip(azimuth):
    if azimuth < 180:
        azimuth = 360- azimuth
    else:
        azimuth = 360 - azimuth
    return azimuth
  
  
def get_camera_matrices_flip(metadata_filename, id):
    # Adaptation of Code/Utils from DISN
    with open(metadata_filename, 'r') as f:
        lines = f.read().splitlines()
        param_lst = read_params(lines)
        rot_mat = get_rotate_matrix(-np.pi / 2)
        az, el, distance_ratio = azimuth_convert_flip(float(param_lst[id][0])), float(param_lst[id][1]), float(param_lst[id][3])
#        print("flip:",az, el, distance_ratio)
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
        self.npyfiles =  get_instance_filenames(data_source, split)

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        mesh_name = self.npyfiles[idx].split(".npz")[0]

        # fetch sdf samples
        sdf_filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        sdf_samples = unpack_sdf_samples(sdf_filename,  self.subsample)

        if self.is_train:
            # reset seed for random sampling training data (see https://github.com/pytorch/pytorch/issues/5059)
            np.random.seed( int(time.time()) + idx)
            id = np.random.randint(0, self.num_views)
        else:
            np.random.seed(idx)
            id = np.random.randint(0, self.num_views)

        view_id = '{0:02d}'.format(id)
#        view_id = 0
#        view_id = str(id)

        image_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), "sketches", view_id + ".png")
#        image_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), "sketches/render_" + view_id + ".png")
        image = unpack_images(image_filename)

        # fetch cameras
        metadata_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), "rendering_metadata.txt")
        intrinsic, extrinsic = get_camera_matrices(metadata_filename, id)
        intrinsic_flip, extrinsic_flip = get_camera_matrices_flip(metadata_filename, id)
        return sdf_samples, image, intrinsic, extrinsic, mesh_name, id, intrinsic_flip, extrinsic_flip
        
#        return sdf_samples, image, intrinsic, extrinsic, mesh_name, id

#        return sdf_samples, image, mesh_name
