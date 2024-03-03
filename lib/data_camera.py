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
import soft_renderer as sr
import soft_renderer.functional as srf

from lib.utils import *

def view2camera(view):
    """
    Caculate camera position from given elevation and azimuth angle.
    The camera looks at the center of the object, with a distance of 2.
    """
    N = view.shape[0]
    distance = torch.ones(N, dtype=torch.float32) * 2.
    camera = srf.get_points_from_angles(distance, view[:, 0], view[:, 1])
    return camera

def get_view_tensor(elevation, azimuth):
    return torch.tensor([elevation], dtype=torch.float32), torch.tensor([azimuth], dtype=torch.float32)


def getview(path, id):
    with open(os.path.join(path, 'view.txt')) as f:
        obj_cameras = [list(map(float, c.split(' '))) for c in list(filter(None, f.read().split('\n')))]
    obj_camera = obj_cameras[id]
    elevation, azimuth = get_view_tensor(obj_camera[1], obj_camera[0])
    view = torch.cat([elevation.unsqueeze(0), azimuth.unsqueeze(0)], dim=1)
    return view


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

        image_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"),"sketches", view_id + ".png")
        RGBA = unpack_images(image_filename)

        # fetch cameras
        # metadata_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), "rendering_metadata.txt")
        # intrinsic, extrinsic = get_camera_matrices(metadata_filename, id)

        view_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"))
        view_sr = getview(view_filename, id)
        camera_sr = view2camera(view_sr)



        return sdf_samples, RGBA, camera_sr, mesh_name
