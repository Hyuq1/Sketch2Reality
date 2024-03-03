import os
import torch
import soft_renderer as sr
import soft_renderer.functional as srf
from scipy.io import loadmat
import numpy as np
import trimesh
import skimage
import torch

import soft_renderer as sr
import soft_renderer.functional as srf
from pytorch3d.loss import chamfer_distance
import skimage.measure
from pytorch3d.io import load_ply
from pytorch3d.io import load_obj
import plyfile
import logging



path = ''
sk_path = ''
dp_path = ''
path_gt = ''


cd_sk = []
cd_re = []
cd_dp = []
iu_sk = []
iu_re = []
iu_dp = []
iu_origin = []
iu_no_score = []

torch.manual_seed(42)

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
    try:
        numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.cpu().detach().numpy()
    except:
        numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor

    # verts, faces, normals, values = skimage.measure.marching_cubes(
    #     numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    # )
    verts, faces, normals, values  = skimage.measure.marching_cubes_lewiner( numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3)

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

def write_obj(verts, faces, obj_path):
    assert obj_path[-4:] == '.obj'
    with open(obj_path, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces+1:
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

for root, dirs, files in os.walk(path):
    rootlist = root.split('/')
    for file in files:
        if (file == 'refined.ply'):
            name = root.split('/')[-1]
            gt_name = name.split('_')[0]

            # sk_name = os.path.join(path_sk, name + '.obj')
            # dp_name = os.path.join(path_dp, name + '.obj')

            mesh_refine_path = os.path.join(root, file)
            mesh_refine = trimesh.load(mesh_refine_path)
            out_vertices = torch.tensor(mesh_refine.vertices).float().cuda()

            scale = (out_vertices.max(axis=0).values - out_vertices.min(axis=0).values).max().item()
            out_vertices = out_vertices / scale * 0.5
            # out_vertices = out_vertices / 2

            re_verts = torch.zeros_like(out_vertices)
            re_verts[:, 0] = out_vertices[:, 0]
            re_verts[:, 1] = out_vertices[:, 1]
            re_verts[:, 2] = out_vertices[:, 2] * -1
            out_vertices = re_verts

            re_verts = out_vertices
            out_faces = torch.tensor(mesh_refine.faces).float().cuda()
            re_faces = out_faces
            faces = srf.face_vertices(out_vertices.unsqueeze(0), out_faces.unsqueeze(0)) * 31. / 32. + 0.5
            faces = faces.cuda().contiguous()
            voxel_re = srf.voxelization(faces, 32, False).cpu().numpy().transpose(0, 2, 1, 3)[..., ::-1]
            # temp1 = torch.tensor(voxel_re.astype(float), requires_grad=True, dtype=torch.float32).squeeze(0)
            # verts_re, faces_re = convert_sdf_samples_to_mesh(temp2.transpose(0, 1).transpose(0, 2), [-0.5, -0.5, -0.5],
            #                                                  1.0 / 32)
            # verts_re, faces_re = convert_sdf_samples_to_mesh(temp1, [-0.5, -0.5, -0.5],
            #                                                  1.0 / 32)
            # verts_re = torch.tensor(verts_re.astype(float), requires_grad=True, dtype=torch.float32)
            # verts_re = verts_re.unsqueeze(0)

            #sk
            # mesh_sk = trimesh.load(sk_name)
            # sk_vert = mesh_sk.vertices
            #
            # scale = (sk_vert.max(axis=0) - sk_vert.min(axis=0)).max().item()
            # out_vertices = sk_vert / scale * 0.5
            # out_vertices = torch.tensor(out_vertices).float().cuda()
            #
            #
            # sk_verts = torch.zeros_like(out_vertices)
            # sk_verts[:, 0] = out_vertices[:, 2]*-1
            # sk_verts[:, 1] = out_vertices[:, 1]
            # sk_verts[:, 2] = out_vertices[:, 0]
            # out_vertices = sk_verts
            # out_faces = torch.tensor(mesh_sk.faces).float().cuda()
            # faces = srf.face_vertices(out_vertices.unsqueeze(0), out_faces.unsqueeze(0)) * 31. / 32. + 0.5
            # faces = faces.cuda().contiguous()
            # voxel_sk = srf.voxelization(faces, 32, False).cpu().numpy().transpose(0, 2, 1, 3)[..., ::-1]
            # sk_faces = out_faces
            #meshsdf_origin
            # origin_path = os.path.join(path_origin, rootlist[-1], 'refined.ply')
            # # origin_path = os.path.join(path_origin, rootlist[-1].split('_')[0] + '_0', 'refined.ply')
            # if not os.path.exists(origin_path):
            #     print("这个文件没有", origin_path)
            #     break
            # mesh_origin = trimesh.load(origin_path)
            # out_vertices = torch.tensor(mesh_origin.vertices).float().cuda()
            #
            # scale = (out_vertices.max(axis=0).values - out_vertices.min(axis=0).values).max().item()
            # out_vertices = out_vertices / scale * 0.5
            # # out_vertices = out_vertices / 2
            #
            # origin_verts = torch.zeros_like(out_vertices)
            # origin_verts[:, 0] = out_vertices[:, 0]
            # origin_verts[:, 1] = out_vertices[:, 1]
            # origin_verts[:, 2] = out_vertices[:, 2] * -1
            # out_vertices = origin_verts
            #
            # origin_verts = out_vertices
            # out_faces = torch.tensor(mesh_origin.faces).float().cuda()
            # origin_faces = out_faces
            # faces = srf.face_vertices(out_vertices.unsqueeze(0), out_faces.unsqueeze(0)) * 31. / 32. + 0.5
            # faces = faces.cuda().contiguous()
            # voxel_origin = srf.voxelization(faces, 32, False).cpu().numpy().transpose(0, 2, 1, 3)[..., ::-1]

            #meshsdf_no_score
            # no_score_path = os.path.join(path_vert_sym, rootlist[-1].split('_')[0], 'refined.ply')
            # # no_score_path = os.path.join(path_vert_sym, rootlist[-1], 'refined.ply')
            # mesh_no_score = trimesh.load(no_score_path)
            # out_vertices = torch.tensor(mesh_no_score.vertices).float().cuda()
            #
            # scale = (out_vertices.max(axis=0).values - out_vertices.min(axis=0).values).max().item()
            # out_vertices = out_vertices / scale * 0.5
            # # out_vertices = out_vertices / 2
            #
            # no_score_verts = torch.zeros_like(out_vertices)
            # no_score_verts[:, 0] = out_vertices[:, 0]
            # no_score_verts[:, 1] = out_vertices[:, 1]
            # no_score_verts[:, 2] = out_vertices[:, 2] * -1
            # out_vertices = no_score_verts
            #
            # no_score_verts = out_vertices
            # out_faces = torch.tensor(mesh_no_score.faces).float().cuda()
            # on_score_faces = out_faces
            # faces = srf.face_vertices(out_vertices.unsqueeze(0), out_faces.unsqueeze(0)) * 31. / 32. + 0.5
            # faces = faces.cuda().contiguous()
            # voxel_no_score = srf.voxelization(faces, 32, False).cpu().numpy().transpose(0, 2, 1, 3)[..., ::-1]


            #dp
            # mesh_dp = trimesh.load(dp_name)
            # dp_vert = mesh_dp.vertices
            #
            # scale = (dp_vert.max(axis=0) - dp_vert.min(axis=0)).max().item()
            # out_vertices = dp_vert / scale * 0.5
            # out_vertices = torch.tensor(out_vertices).float().cuda()
            #
            #
            # dp_verts = torch.zeros_like(out_vertices)
            # dp_verts[:, 0] = out_vertices[:, 2] * -1
            # dp_verts[:, 1] = out_vertices[:, 1]
            # dp_verts[:, 2] = out_vertices[:, 0]
            # out_vertices = dp_verts
            # out_faces = torch.tensor(mesh_dp.faces).float().cuda()
            # faces = srf.face_vertices(out_vertices.unsqueeze(0), out_faces.unsqueeze(0)) * 31. / 32. + 0.5
            # faces = faces.cuda().contiguous()
            # voxel_dp = srf.voxelization(faces, 32, False).cpu().numpy().transpose(0, 2, 1, 3)[..., ::-1]
            # dp_faces = out_faces



            # temp2 = torch.tensor(voxel_sk.astype(float), requires_grad=True, dtype=torch.float32).squeeze(0)
            # # verts_re, faces_re = convert_sdf_samples_to_mesh(temp2.transpose(0, 1).transpose(0, 2), [-0.5, -0.5, -0.5],
            # #                                                  1.0 / 32)
            # verts_sk, faces_sk = convert_sdf_samples_to_mesh(temp2.transpose(1, 2), [-0.5, -0.5, -0.5],
            #                                                  1.0 / 32)
            # verts_sk = torch.tensor(verts_sk.astype(float), requires_grad=True, dtype=torch.float32)
            # verts_sk = verts_sk.unsqueeze(0)

            path1 = os.path.join(path_gt, gt_name + '.obj')
            mesh_gt = trimesh.load(path1, force='mesh')
            gt_verts = torch.tensor(mesh_gt.vertices).float().cuda()

            scale = (gt_verts.max(axis=0).values - gt_verts.min(axis=0).values).max().item()
            gt_verts = gt_verts / scale * 0.5

            gt_faces = torch.tensor(mesh_gt.faces).float().cuda()
            faces = srf.face_vertices(gt_verts.unsqueeze(0), gt_faces.unsqueeze(0)) * 31. / 32. + 0.5
            faces = faces.cuda().contiguous()
            voxel_gt = srf.voxelization(faces, 32, False).cpu().numpy().transpose(0, 2, 1, 3)[..., ::-1]
            # temp = torch.tensor(voxel_gt.astype(float), requires_grad=True, dtype=torch.float32).squeeze(0)
            # verts_gt, faces_gt = convert_sdf_samples_to_mesh(temp, [-0.5, -0.5, -0.5],
            #                                                  1.0 / 32)
            # v, f = verts_gt, faces_gt


            # verts_gt = torch.tensor(verts_gt.astype(float), requires_grad=True, dtype=torch.float32)
            # verts_gt = verts_gt.unsqueeze(0).cuda()

            chd_re, _ = chamfer_distance(gt_verts.unsqueeze(0).cuda(), re_verts.unsqueeze(0))
            # chd_sk, _ = chamfer_distance(gt_verts.unsqueeze(0).cuda(), sk_verts.unsqueeze(0))
            # chd_dp, _ = chamfer_distance(gt_verts.unsqueeze(0).cuda(), dp_verts.unsqueeze(0))
            chd_re = chd_re * 1000
            # chd_sk = chd_sk * 1000
            # chd_dp = chd_dp * 1000
            cd_re.append(chd_re)
            # cd_sk.append(chd_sk)
            # cd_dp.append(chd_dp)

            # verts_re, faces_re = convert_sdf_samples_to_mesh(voxel_re[0], [-1, -1, -1], 2.0 / 31)
            # verts_gt, faces_gt = convert_sdf_samples_to_mesh(voxel_gt[0], [-1, -1, -1], 2.0 / 31)
            # verts_sk, faces_sk = convert_sdf_samples_to_mesh(voxel_sk[0], [-1, -1, -1], 2.0 / 31)
            # write_obj(verts_gt, faces_gt, 'gt.obj')
            # write_obj(verts_re, faces_re, 're.obj')
            # write_obj(verts_sk, faces_sk, 'dp.obj')
            # #
            # write_obj(gt_verts, gt_faces, 'mgt.obj')
            # write_obj(re_verts, re_faces, 'mre.obj')
            # write_obj(sk_verts, sk_faces, 'msk.obj')


            iou_ms = voxel_iou(voxel_re, voxel_gt)
            iu_re.append(iou_ms)
            # iou_origin = voxel_iou(voxel_origin, voxel_gt)
            # iu_origin.append(iou_origin)
            # iou_no_score = voxel_iou(voxel_no_score, voxel_gt)
            # iu_no_score.append(iou_no_score)
            # iou_sk = voxel_iou(voxel_sk, voxel_gt)
            # iu_sk.append(iou_sk)
            # iou_dp = voxel_iou(voxel_dp, voxel_gt)
            # iu_dp.append(iou_dp)
            # print('chd_re:', chd_re, 'chd_sk:', chd_sk)
            # print('iou_ms:', iou_ms, 'iou_sk:', iou_sk)

s1 = 0
s2 = 0
s3 = 0
s4 = 0
s5 = 0
s6 = 0
s7 = 0
# for x in cd_re:
#     s1 += x
# for x in cd_sk:
#     s2 += x

for x in iu_re:
    s3 += x
# for x in iu_origin:
#     s4 += x
# for x in iu_no_score:
#     s5 += x

# for x in iu_sk:
#     s6 += x
#
# for x in iu_dp:
#     s7 += x




# re_mean = s1 / len(cd_re)
# sk_mean = s2 / len(cd_sk)
iou_score_mean = s3 / len(iu_re)
# iou_origin_mean = s4 / len(iu_origin)
# iou_v_mean = s5 / len(iu_no_score)
# iou_dp_mean = s7 / len(iu_dp)
# iou_sk_mean = s6 / len(iu_sk)
# print('chd_re_mean:', re_mean, 'chd_sk_mean:', sk_mean, 'voxel_iou_ms:', iou_ms_mean, 'voxel_iou_sk:', iou_sk_mean)
# print('iou_ms:', iou_ms_mean, 'iou_dp:', iou_sk_mean, 'iou_sk:', iou_dp_mean)
# print('iou_ms:', iou_score_mean, 'iou_no_score:', iou_v_mean, 'iou_origin:', iou_origin_mean)
# print('meshsdf:', iou_score_mean, 'deep3d:', iou_dp_mean, 'sk:', iou_sk_mean)
print('meshsdf:', iou_score_mean)

