import os
import shutil
import pytorch3d.loss
import scipy
import torch
import soft_renderer as sr
import soft_renderer.functional as srf
from scipy.io import loadmat
import numpy as np
import trimesh
import skimage
from pytorch3d.loss import chamfer_distance
import skimage.measure
import plyfile
import logging
from pytorch3d.io import load_ply, load_obj
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.ops import sample_points_from_meshes
import numpy as np
import tqdm
import open3d as o3d




path_origin = ''
path = ''
path_vert_sym = ''
path_gt = ''


cd_dp = []
cd_re = []
cd_sk = []
voxel_sum = []
cd_re_v_sym = []
cd_re_v_i_sym = []
cd_origin = []

torch.manual_seed(42)

def write_verts_faces_to_file(verts, faces, ply_filename_out):

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


def voxel_iou(pred, target):
    return ((pred * target).sum((1, 2, 3)) / (0 < (pred + target)).sum((1, 2, 3))).mean()


def convert_sdf_samples_to_mesh(pytorch_3d_sdf_tensor,
                                voxel_grid_origin,
                                voxel_size,
                                offset=None,
                                scale=None,
                                ):
    """
    Convert sdf samples to verts, faces

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.cpu().detach().numpy()

    # verts, faces, normals, values = skimage.measure.marching_cubes(
    #     numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    # )
    verts, faces, normals, values  = skimage.measure.marching_cubes( numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3)

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = (voxel_grid_origin[2] + verts[:, 2]) * -1

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # return mesh
    return mesh_points, faces

def convert_sdf_samples_to_mesh1(pytorch_3d_sdf_tensor,
                                voxel_grid_origin,
                                voxel_size,
                                offset=None,
                                scale=None,
                                ):
    """
    Convert sdf samples to verts, faces

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.cpu().detach().numpy()

    # verts, faces, normals, values = skimage.measure.marching_cubes(
    #     numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    # )
    verts, faces, normals, values  = skimage.measure.marching_cubes( numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3)

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # return mesh
    return mesh_points, faces


def sample_ply(obj_paths):
    mesh_list = []
    for obj_path in obj_paths:
        if not os.path.exists(obj_path):
            obj_path = os.path.join(os.path.dirname(__file__), 'template.ply')
        verts, faces = load_ply(obj_path)
        mesh = Meshes(verts=[verts], faces=[faces])
        mesh_list.append(mesh)
    meshes = join_meshes_as_batch(mesh_list)
    pcs = sample_points_from_meshes(
                meshes, num_samples=4096)
    return pcs

def sample_obj(obj_paths):
    mesh_list = []
    for obj_path in obj_paths:
        if not os.path.exists(obj_path):
            obj_path = os.path.join(os.path.dirname(__file__), 'template.ply')
        verts, faces, _ = load_obj(obj_path)
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
        mesh_list.append(mesh)
    meshes = join_meshes_as_batch(mesh_list)
    pcs = sample_points_from_meshes(
                meshes, num_samples=4096)
    return pcs


def normalize_to_box(input):
    """
    normalize point cloud to unit bounding box
    center = (max - min)/2
    scale = max(abs(x))
    input: pc [N, P, dim] or [P, dim]
    output: pc, centroid, furthest_distance

    From https://github.com/yifita/pytorch_points
    """
    if len(input.shape) == 2:
        axis = 0
        P = input.shape[0]
        D = input.shape[1]
    elif len(input.shape) == 3:
        axis = 1
        P = input.shape[1]
        D = input.shape[2]
    else:
        raise ValueError()

    if isinstance(input, np.ndarray):
        maxP = np.amax(input, axis=axis, keepdims=True)
        minP = np.amin(input, axis=axis, keepdims=True)
        centroid = (maxP + minP) / 2
        input = input - centroid
        furthest_distance = np.amax(np.abs(input), axis=(axis, -1), keepdims=True)
        input = input / furthest_distance
    elif isinstance(input, torch.Tensor):
        maxP = torch.max(input, dim=axis, keepdim=True)[0]
        minP = torch.min(input, dim=axis, keepdim=True)[0]
        centroid = (maxP + minP) / 2
        input = input - centroid
        in_shape = list(input.shape[:axis]) + [P * D]
        furthest_distance = torch.max(torch.abs(input).reshape(in_shape), dim=axis, keepdim=True)[0]
        furthest_distance = furthest_distance.unsqueeze(-1)
        input = input / furthest_distance
    else:
        raise ValueError()

    return input, centroid, furthest_distance



for root, dirs, files in os.walk(path):
    rootlist = root.split('/')
    for file in files:
        if(file == 'refined.ply'):
            mesh_refine_path = os.path.join(root, file)
            mesh_refine = trimesh.load(mesh_refine_path)

            gen_pc = sample_ply([mesh_refine_path])[0]
            gen_pc = normalize_to_box(gen_pc)[0]


            mesh_name = rootlist[-1].split('_')[0]

            #meshsdf vert sym
            # ms_v_sym_path = os.path.join(path_vert_sym, rootlist[-1].split('_')[0], 'refined.ply')
            ms_v_sym_path = os.path.join(path_vert_sym, rootlist[-1], 'refined.ply')
            ms_v_sym_pc = sample_ply([ms_v_sym_path])[0]
            ms_v_sym_pc = normalize_to_box(ms_v_sym_pc)[0]

            #meshsdf_origin
            origin_path = os.path.join(path_origin, rootlist[-1], 'refined.ply')
            if not os.path.exists(origin_path):
                print("这个文件没有", origin_path)
                break
            ms_origin_pc = sample_ply([origin_path])[0]
            ms_origin_pc = normalize_to_box(ms_origin_pc)[0]

            #meshsdf vert img sym
            # ms_v_i_sym_path = os.path.join(path_vert_img_sym, rootlist[-1], 'refined.ply')
            # ms_v_i_sym_pc = sample_ply([ms_v_i_sym_path])[0]
            # ms_v_i_sym_pc = normalize_to_box(ms_v_i_sym_pc)[0]
#sketch2model
            # mesh_sk_path = os.path.join(path_sk, rootlist[-1] + '.obj')
            # sk_pc = sample_obj([mesh_sk_path])[0]
            # ck_pc = torch.zeros_like(sk_pc)
            # ck_pc[:, 0] = sk_pc[:, 2] * -1
            # ck_pc[:, 1] = sk_pc[:, 1]
            # ck_pc[:, 2] = sk_pc[:, 0]
            # sk_pc = ck_pc
            # sk_pc = normalize_to_box(sk_pc)[0]

#deep3d
            # mesh_dp_path = os.path.join(path_dp, rootlist[-1] + '.obj')
            # dp_pc = sample_obj([mesh_dp_path])[0]
            # c_pc = torch.zeros_like(dp_pc)
            # c_pc[:, 0] = dp_pc[:, 2] * -1
            # c_pc[:, 1] = dp_pc[:, 1]
            # c_pc[:, 2] = dp_pc[:, 0]
            # dp_pc = c_pc
            # dp_pc = normalize_to_box(dp_pc)[0]
#gt
            path1 = os.path.join(path_gt, mesh_name + '.obj')
            gt_pc = sample_obj([path1])[0]
            gt_pc = normalize_to_box(gt_pc)[0]
            #
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud1 = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(gen_pc.numpy())
            # point_cloud1.points = o3d.utility.Vector3dVector(gt_pc.numpy())
            #
            # colors = np.array([255, 0, 0])  # RGB颜色值
            # point_cloud.paint_uniform_color(colors)
            # colors = np.array([0, 0, 255])  # RGB颜色值
            # point_cloud1.paint_uniform_color(colors)
            # o3d.visualization.draw_geometries([point_cloud, point_cloud1])


            chd_re = pytorch3d.loss.chamfer_distance(gt_pc[None], gen_pc[None], batch_reduction=None)[0]
            # chd_sk = pytorch3d.loss.chamfer_distance(gt_pc[None], sk_pc[None], batch_reduction=None)[0]
            # chd_dp = pytorch3d.loss.chamfer_distance(gt_pc[None], dp_pc[None], batch_reduction=None)[0]
            chd_re_v_sym = pytorch3d.loss.chamfer_distance(gt_pc[None], ms_v_sym_pc[None], batch_reduction=None)[0]
            # # chd_re_v_i_sym = pytorch3d.loss.chamfer_distance(gt_pc[None], ms_v_i_sym_pc[None], batch_reduction=None)[0]
            chd_origin = pytorch3d.loss.chamfer_distance(gt_pc[None], ms_origin_pc[None], batch_reduction=None)[0]
            chd_re = chd_re * 1000
            # chd_sk = chd_sk * 1000
            # chd_dp = chd_dp * 1000
            chd_re_v_sym = chd_re_v_sym *1000
            chd_origin = chd_origin * 1000
            # chd_re_v_i_sym = chd_re_v_i_sym * 1000
            cd_re.append(chd_re)
            # cd_sk.append(chd_sk)
            cd_origin.append(chd_origin)
            # # cd_dp.append(chd_dp)
            cd_re_v_sym.append(chd_re_v_sym)
            # cd_re_v_i_sym.append(chd_re_v_i_sym)


            # voxel_iou_ms = voxel_iou(voxel_re, gt_v)
            # voxel_sum.append(voxel_iou_ms)
            # print('chd_re:', chd_re, 'chd_sk:', chd_sk, 'chd_dp:', chd_dp)
            # print('chd_re_score:', chd_re)





s1 = 0
s2 = 0
s3 = 0
s4 = 0
s5 = 0
# for x in voxel_sum:
#     s1 += x

for x in cd_re:
    s1 += x
# for x1 in cd_sk:
#     s2 += x1
#
# for x2 in cd_dp:
#     s3 += x2

for x4 in cd_re_v_sym:
    s4 += x4

for x5 in cd_origin:
    s5 += x5

re_mean = s1/len(cd_re)
# sk_mean = s2/len(cd_sk)
# dp_mean = s3/len(cd_dp)
re_v_mean = s4/len(cd_re_v_sym)
re_origin_mean = s5/len(cd_origin)
# iou_mean = s1/len(voxel_sum)
# print('chd_re_mean:', re_mean, 'chd_sk_mean:', sk_mean, 'chd_dp_mean:', dp_mean)
# print('chd_re_mean:', re_mean, 'chd_re_v_mean:', re_v_mean, 'chd_re_v_i_mean:', re_v_i_mean)
# print('chd_re_mean:', re_mean, 'chd_re_v_mean:', re_v_mean)
print('chd_v_score_mean:', re_mean, 'chd_v_no_score_mean:', re_v_mean, 'chd_origin_mean:', re_origin_mean)
# print('chd_v_score_mean:', re_mean)
