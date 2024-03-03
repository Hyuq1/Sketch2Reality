#!/usr/bin/env python3

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import json
import time
import pdb

import imageio
import numpy as np
from sklearn.neighbors import NearestNeighbors

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    SoftSilhouetteShader,
    TexturesVertex
)
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes

import lib
import lib.workspace as ws
from lib.utils import *
from models.deformation_net_graph_convolutional_lite import DeformationNetworkGraphConvolutionalLite




def vertex_symmetry_loss(mesh, sym_plane, device, asym_conf_scores=False, sym_bias=0.005):
    """Vertex based symmetry loss.

    Args:
        mesh (Mesh): pytorch3d mesh object
        sym_plane (torch.tensor): vector orthogonal to the plane of symmetry
        device (torch.device): pytorch device
        asym_conf_scores (bool, optional): If asymmetry confidence scores should be used. Defaults to False.
        sym_bias (float, optional): Bias term for symmetry. Defaults to 0.005.

    Raises:
        ValueError: Symmetry plane vector needs to be a unit normal vector.

    Returns:
        torch.tensor: loss value
    """
    N = np.array([sym_plane])
    if np.linalg.norm(N) != 1:
        raise ValueError("sym_plane needs to be a unit normal")

    reflect_matrix = torch.tensor(np.eye(3) - 2 * N.T @ N, dtype=torch.float).to(device)
    mesh_verts = mesh.verts_packed()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(mesh_verts.detach().cpu())
    sym_points = mesh_verts @ reflect_matrix
    distances, indices = nbrs.kneighbors(sym_points.detach().cpu())
    nn_dists = torch.unsqueeze(
        torch.sum(torch.nn.functional.mse_loss(sym_points, torch.squeeze(mesh_verts[indices], 1), reduction='none'), 1), 1)

    if asym_conf_scores is not None:
        avg_sym_loss = torch.mean(
            torch.log(torch.div(1, asym_conf_scores)) * sym_bias + ((asym_conf_scores) * nn_dists))
    else:
        avg_sym_loss = torch.mean(nn_dists)

    return avg_sym_loss


def main_function(experiment_directory, continue_from,  iterations, marching_cubes_resolution, regularize):

    device=torch.device('cuda:0')
#    torch.cuda.set_device(5)
    specs = ws.load_experiment_specifications(experiment_directory)

    print("Reconstruction from experiment description: \n" + ' '.join([str(elem) for elem in specs["Description"]]))

    data_source = specs["DataSource"]
    class_id = specs["class_id"]
    test_split_file = specs["TestSplit"]

    arch_encoder = __import__("lib.models." + specs["NetworkEncoder"], fromlist=["ResNet"])
    arch_decoder = __import__("lib.models." + specs["NetworkDecoder"], fromlist=["DeepSDF"])
    latent_size = specs["CodeLength"]

    encoder = arch_encoder.ResNet(latent_size, specs["Depth"], norm_type = specs["NormType"]).cuda()
    decoder = arch_decoder.DeepSDF(latent_size, **specs["NetworkSpecs"]).cuda()

    encoder = torch.nn.DataParallel(encoder)
    decoder = torch.nn.DataParallel(decoder)
    
    deform_net = DeformationNetworkGraphConvolutionalLite(device)
    deform_net.to(device)

    print("testing with {} GPU(s)".format(torch.cuda.device_count()))

    num_samp_per_scene = specs["SamplesPerScene"]
    with open(test_split_file, "r") as f:
        test_split = json.load(f)

#    sdf_dataset_test = lib.data.RGBA2SDF(
#        data_source, test_split, num_samp_per_scene, is_train=False, num_views = specs["NumberOfViews"]
#    )
    sdf_dataset_test = lib.data_new.RGBA2SDF(
        data_source, class_id, test_split, num_samp_per_scene, is_train=False, num_views = specs["NumberOfViews"]
    )
    
    torch.manual_seed(int( time.time() * 1000.0 ))
    sdf_loader_test = data_utils.DataLoader(
        sdf_dataset_test,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=False,
    )

    num_scenes = len(sdf_loader_test)
    print("There are {} scenes".format(num_scenes))

    print('Loading epoch "{}"'.format(continue_from))
    
   

    ws.load_model_parameters(
        experiment_directory, continue_from, encoder, decoder
    )
    encoder.eval()

    optimization_meshes_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(continue_from)
    )

    if not os.path.isdir(optimization_meshes_dir):
        os.makedirs(optimization_meshes_dir)

#    for sdf_data, image, intrinsic, extrinsic, name, id, intrinsic_flip, extrinsic_flip in sdf_loader_test:
    for sdf_data, image, intrinsic, extrinsic, name in sdf_loader_test:

        out_name = name[0].split("/")[-1]
#        number = id.item()
#        out_name = out_name + f'_{number}'
        # store input stuff
        image_filename = os.path.join(optimization_meshes_dir, out_name, "input.png")
        # skip if it is already there
        if os.path.exists(os.path.dirname(image_filename)):
            print(name[0], " exists already ")
            continue
        print('Reconstructing {}...'.format(out_name))

        if not os.path.exists(os.path.dirname(image_filename)):
            os.makedirs(os.path.dirname(image_filename))

        image_export = 255*image[0].permute(1,2,0).cpu().numpy()
        imageio.imwrite(image_filename, image_export.astype(np.uint8))

        # image_filename = os.path.join(optimization_meshes_dir, out_name, "input_silhouette.png")
        # image_export = 255*image[0].permute(1,2,0).cpu().numpy()[...,3]
        # imageio.imwrite(image_filename, image_export.astype(np.uint8))

        # get latent code from image
#        import pdb
#        pdb.set_trace()
#        image = image.cuda()
        latent = encoder(image)
        # get estimated mesh
        verts, faces, samples, next_indices = lib.mesh.create_mesh(decoder, latent, N=marching_cubes_resolution, output_mesh = True)

        # store raw output
        mesh_filename = os.path.join(optimization_meshes_dir, out_name, "predicted.ply")
        lib.mesh.write_verts_faces_to_file(verts, faces, mesh_filename)

        verts_dr = torch.tensor(verts[None, :, :].copy(), dtype=torch.float32, requires_grad = False).cuda()
        verts_dr = verts_dr / 2
        faces_dr = torch.tensor(faces[None, :, :].copy()).cuda()

        IMG_SIZE = image.shape[-1]
        K_cuda = torch.tensor(intrinsic[:, 0:3, 0:3]).float().cuda()
        R_cuda = torch.tensor(extrinsic[:, 0:3, 0:3]).float().cuda().permute(0,2,1)
        t_cuda = torch.tensor(extrinsic[:, 0:3, 3]).float().cuda()
        lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
        cameras = PerspectiveCameras(device=device, focal_length=-K_cuda[:,0,0] / K_cuda[:,0,2], image_size=((IMG_SIZE, IMG_SIZE),), R=R_cuda, T=t_cuda)
        #flip cameras
#        K_cuda_flip = torch.tensor(intrinsic_flip[:, 0:3, 0:3]).float().cuda()
#        R_cuda_flip = torch.tensor(extrinsic_flip[:, 0:3, 0:3]).float().cuda().permute(0, 2, 1)
#        t_cuda_flip = torch.tensor(extrinsic_flip[:, 0:3, 3]).float().cuda()
        # lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
#        cameras_flip = PerspectiveCameras(device=device, focal_length=-K_cuda_flip[:, 0, 0] / K_cuda_flip[:, 0, 2],
#                                     image_size=((IMG_SIZE, IMG_SIZE),), R=R_cuda_flip, T=t_cuda_flip)
        
        
        
        
        raster_settings = RasterizationSettings(
            image_size=IMG_SIZE,
            blur_radius=0.000001,
            faces_per_pixel=1,
            bin_size=0,
        )
        raster_settings_soft = RasterizationSettings(
            image_size=IMG_SIZE,
            blur_radius=np.log(1. / 1e-4 - 1.)*1e-5,
            faces_per_pixel=25,
        )

        # instantiate renderers
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings_soft
            ),
            shader=SoftSilhouetteShader()
        )
        depth_renderer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        #flip instantiate renderers
#        silhouette_renderer_flip = MeshRenderer(
#            rasterizer=MeshRasterizer(
#                cameras=cameras_flip,
#                raster_settings=raster_settings_soft
#            ),
#            shader=SoftSilhouetteShader()
#        )
#        depth_renderer_flip = MeshRasterizer(
#            cameras=cameras_flip,
#            raster_settings=raster_settings
#        )
        
        
        
        
        renderer = Renderer(silhouette_renderer, depth_renderer, image_size=IMG_SIZE)
#        renderer_flip = Renderer(silhouette_renderer_flip, depth_renderer_flip, image_size=IMG_SIZE)
        
        

        meshes = Meshes(verts_dr, faces_dr)
        verts_shape = meshes.verts_packed().shape
        verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=False)
        meshes.textures = TexturesVertex(verts_features=verts_rgb)

        with torch.no_grad():


            normal_out, silhouette_out = renderer(meshes_world=meshes, cameras=cameras, lights=lights)

            image_out_export = 255*silhouette_out.detach().cpu().numpy()[0]
            image_out_filename = os.path.join(optimization_meshes_dir, out_name, "predicted_silhouette.png")
            imageio.imwrite(image_out_filename, image_out_export.astype(np.uint8))

            image_out_export = 255*normal_out.detach().cpu().numpy()[0]
            image_out_filename = os.path.join(optimization_meshes_dir, out_name, "predicted.png")
            imageio.imwrite(image_out_filename, image_out_export.astype(np.uint8))


          
        

        latent_for_optim = torch.tensor(latent, requires_grad = True)
        lr= 5e-5
        optimizer = torch.optim.Adam([latent_for_optim], lr=lr)

        decoder.eval()

        log_silhouette = []
        log_latent = []
        log_chd = []
        log_nc = []

        for e in range(iterations+1):

            optimizer.zero_grad()

            # first create mesh
            verts, faces, samples, next_indices = lib.mesh.create_mesh_optim_fast(samples, next_indices, decoder, latent_for_optim, N=marching_cubes_resolution)

            # now assemble loss function
            verts = verts / 2
            xyz_upstream = torch.tensor(verts.astype(float), requires_grad = True, dtype=torch.float32, device=device)
            faces_upstream = torch.tensor(faces.astype(float), requires_grad = False, dtype=torch.float32, device=device)
            

            meshes_dr = Meshes(xyz_upstream.unsqueeze(0), faces_upstream.unsqueeze(0))
            verts_shape = meshes_dr.verts_packed().shape
            verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=False)
            meshes_dr.textures = TexturesVertex(verts_features=verts_rgb)

            normal, silhouette = renderer(meshes_world=meshes_dr, cameras=cameras, lights=lights)
            
            #对称置信度
            asym_conf_scores = deform_net(image[:, :3, :], meshes_dr, R_cuda, t_cuda)
            
            #flip
#            image_out_export_o = 255 * silhouette.detach().cpu().numpy()[0]
#            image_out_filename = os.path.join(optimization_meshes_dir, out_name, "origin_sil.png")
#            imageio.imwrite(image_out_filename, image_out_export_o.astype(np.uint8))
#
#            normal_flip, silhouette_flip = renderer_flip(meshes_world=meshes_dr, cameras=cameras_flip, lights=lights)
#            silhouette_flip = torch.flip(silhouette_flip, [2])
#            silhouette_flip_1 = torch.flip(silhouette_flip, [1])
            
#            image_out_export_f = 255 * silhouette_flip.detach().cpu().numpy()[0]
#            image_out_filename = os.path.join(optimization_meshes_dir, out_name, "flip_sil.png")
#            imageio.imwrite(image_out_filename, image_out_export_f.astype(np.uint8))
        
#            image_out_export_f = 255 * silhouette_flip.detach().cpu().numpy()[0]
#            image_out_filename = os.path.join(optimization_meshes_dir, out_name, "flip_sil.png")
#            imageio.imwrite(image_out_filename, image_out_export_f.astype(np.uint8))
            
#            image_out_export_f = 255 * silhouette_flip_1.detach().cpu().numpy()[0]
#            image_out_filename = os.path.join(optimization_meshes_dir, out_name, "flip_sil_1.png")
#            imageio.imwrite(image_out_filename, image_out_export_f.astype(np.uint8))
            
            
            
            # compute loss
            loss_silhouette = (torch.abs(silhouette - image[:,3].cuda())).mean()
          
            



            # now store upstream gradients
            loss_silhouette.backward(retain_graph=True)
            dL_dx_i = xyz_upstream.grad
            # take care of weird stuff possibly happening
            dL_dx_i[torch.isnan(dL_dx_i)] = 0

            # log stuff
            with torch.no_grad():
                log_silhouette.append(loss_silhouette.detach().cpu().numpy())

                # meshes_gt_pts = sample_points_from_meshes(meshes_gt)
                meshes_dr_pts = sample_points_from_meshes(meshes_dr)
                # metric_chd, _ = chamfer_distance(meshes_gt_pts, meshes_dr_pts)
                # log_chd.append(metric_chd.detach().cpu().numpy())
                #
                # log_nc.append(compute_normal_consistency(normal_tgt, normal))

                log_latent.append(torch.mean((latent_for_optim).pow(2)).detach().cpu().numpy())



            # use vertices to compute full backward pass
            optimizer.zero_grad()
            xyz = torch.tensor(verts.astype(float), requires_grad = True,dtype=torch.float32, device=torch.device('cuda:0'))
            latent_inputs = latent_for_optim.expand(xyz.shape[0], -1)
            #first compute normals
            pred_sdf = decoder(latent_inputs, xyz)
            loss_normals = torch.sum(pred_sdf)
            loss_normals.backward(retain_graph = True)
            normals = xyz.grad/torch.norm(xyz.grad, 2, 1).unsqueeze(-1)
            # now assemble inflow derivative
            optimizer.zero_grad()
            dL_ds_i = -torch.matmul(dL_dx_i.unsqueeze(1), normals.unsqueeze(-1)).squeeze(-1)
            #对称损失
            loss_symmetry = vertex_symmetry_loss(meshes_dr, [0, 0, 1], device=torch.device("cuda:0"), asym_conf_scores=asym_conf_scores, sym_bias=0.005)
            
            #img sym loss
#            loss_sil_sym = (torch.abs(silhouette - silhouette_flip)).mean()
            # finally assemble full backward pass
            
        


            loss_backward = torch.sum(dL_ds_i * pred_sdf) + regularize * torch.mean((latent_for_optim).pow(2)) + loss_symmetry * 0.1
            loss_backward.backward()
            # and update params
            optimizer.step()

        # store all
        with torch.no_grad():
            verts, faces, samples, next_indices = lib.mesh.create_mesh_optim_fast(samples, next_indices, decoder, latent_for_optim, N=marching_cubes_resolution)
            mesh_filename = os.path.join(optimization_meshes_dir, out_name, "refined.ply")
            lib.mesh.write_verts_faces_to_file(verts, faces, mesh_filename)
            xyz_upstream = torch.tensor(verts.astype(float), requires_grad = True, dtype=torch.float32, device=device)
            xyz_upstream = xyz_upstream / 2
            faces_upstream = torch.tensor(faces.astype(float), requires_grad = False, dtype=torch.float32, device=device)

            meshes_dr = Meshes(xyz_upstream.unsqueeze(0), faces_upstream.unsqueeze(0))
            verts_shape = meshes_dr.verts_packed().shape
            verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=False)
            meshes_dr.textures = TexturesVertex(verts_features=verts_rgb)

            normal, silhouette = renderer(meshes_world=meshes_dr, cameras=cameras, lights=lights)

            image_out_export = 255*silhouette.detach().cpu().numpy()[0]
            # image_out_filename = os.path.join(optimization_meshes_dir, out_name, "refined_silhouette.png")
            # imageio.imwrite(image_out_filename, image_out_export.astype(np.uint8))
            image_out_export = 255*normal.detach().cpu().numpy()[0]
            # image_out_filename = os.path.join(optimization_meshes_dir, out_name, "refined.png")
            # imageio.imwrite(image_out_filename, image_out_export.astype(np.uint8))

        log_filename = os.path.join(optimization_meshes_dir, out_name,  "log_silhouette.npy")
        np.save(log_filename, log_silhouette)
        log_filename = os.path.join(optimization_meshes_dir, out_name, "log_chd.npy")
        np.save(log_filename, log_chd)
        log_filename = os.path.join(optimization_meshes_dir, out_name, "log_nc.npy")
        np.save(log_filename, log_nc)

        # compute_normal_consistency(normal_tgt, normal)

        log_filename = os.path.join(optimization_meshes_dir, out_name, "log_latent.npy")
        np.save(log_filename, log_latent)
        print('Done with refinement.')



if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        default="latest",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--resolution",
        default=256,
        help="Marching cubes resolution for reconstructed surfaces.",
    )
    arg_parser.add_argument(
        "--iterations",
        default=100,
        help="Number of refinement iterations.",
    )
    arg_parser.add_argument("--regularize", default=0.0, help="L2 regularization weight on latent vector")

    args = arg_parser.parse_args()
    main_function(args.experiment_directory, args.continue_from, int(args.iterations), int(args.resolution), float(args.regularize))
