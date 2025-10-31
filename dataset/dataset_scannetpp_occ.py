
from typing import List, Tuple, Dict
import os
import json
import glob
import numpy as np
import numba as nb
import torch
from torch.utils import data
import pickle
from PIL import Image
from mmcv.image.io import imread
import copy
from pyquaternion import Quaternion
from . import OPENOCC_DATASET
from dataset.nyu_utils import vox2pix
from torchvision import transforms
from mmcv.image.io import imread
import math, cv2
from torchvision.transforms import Compose
from dataset.transform_ import Resize, NormalizeImage, PrepareForNet
from mmengine.fileio import load  # for reading ann_file (pkl/json)


@OPENOCC_DATASET.register_module()
class Scannetpp2xDataset(data.Dataset):
    """ScanNet++ dataset (ann_file-driven) that returns OpenOcc-style tuples.

    - Reads a mmengine-style annotation file (dict with "metainfo" and "data_list").
    - Uses parse_ann_info() to load occupancy from disk (as N x 4 point format: x,y,z,label).
    - __getitem__ returns (imgs, meta, occs):
        * imgs: (1, N_views, H, W, 3) float32 RGB in range [0, 255]
        * meta: dict with camera intrinsics/extrinsics and auxiliary fields
        * occs: (N_pts, 4) int64 array directly from parse_ann_info()
    """

    def __init__(self,
                 data_path: str,
                 ann_file: str = 'scannetpp_infos_train.pkl',
                 num_frames: int = 1,
                 offset: int = 0,
                 grid_size_occ: List[int] = [240, 240, 80],
                 coarse_ratio: int = 2,
                 empty_idx: int = 0,
                 phase: str = 'train',
                 num_pts: int = 21600,
                 data_tg: str = 'base'):

        super().__init__()
        self.scannetpp_root = data_path
        self.phase = phase
        self.ann_file = ann_file if os.path.isabs(ann_file) else os.path.join(self.scannetpp_root, ann_file)

        self.num_frames = num_frames
        self.offset = offset
        self.grid_size_occ = grid_size_occ
        self.grid_size_occ_coarse = (np.array(grid_size_occ) // coarse_ratio).astype(np.uint32)
        self.coarse_ratio = coarse_ratio
        self.empty_idx = empty_idx
        self.num_pts = num_pts

        self.voxel_size = 0.05  # 0.05m
        self.scene_size = (12.0, 12.0, 4.0)  # (12.0m, 12.0m, 4.0m)
        self.scene_min_bound = (-6.0, -6.0, -0.78)

        # Holder for mmengine-like metainfo and data prefix
        self._metainfo: Dict = {}
        self.data_prefix = dict(img_path=self.scannetpp_root)

        # Load annotations → build data_list
        self.data_list = self.load_data_list()
        
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # ---------------------------
    # mmengine-like utilities
    # ---------------------------
    def process_metainfo(self) -> None:
        """Build label and occupancy label mappings from metainfo."""
        assert 'categories' in self._metainfo

        if 'classes' not in self._metainfo:
            self._metainfo.setdefault('classes', list(self._metainfo['categories'].keys()))

        # Map raw label id → contiguous class index
        self.label_mapping = np.full(
            max(list(self._metainfo['categories'].values())) + 1, -1, dtype=int)
        for key, value in self._metainfo['categories'].items():
            if key in self._metainfo['classes']:
                self.label_mapping[value] = self._metainfo['classes'].index(key)

        # Map raw label id → occupancy class index (1-based; 0 is empty)
        self.occ_label_mapping = np.full(
            max(list(self._metainfo['categories'].values())) + 1, -1, dtype=int)
        if 'occ_classes' in self._metainfo:
            for idx, label_name in enumerate(self._metainfo['occ_classes']):
                self.occ_label_mapping[self._metainfo['categories'][label_name]] = idx + 1

    def _get_axis_align_matrix(self, info: dict) -> np.ndarray:
        """Return axis_align_matrix or identity if missing."""
        if 'axis_align_matrix' in info:
            return np.array(info['axis_align_matrix'])
        warnings.warn('axis_align_matrix missing; using identity.')
        return np.eye(4).astype(np.float32)

    def parse_ann_info(self, info: dict) -> dict:
        """Load instances and occupancy (kept as point list N x 4)."""
        # 3D boxes and labels (optional)
        ann_info = None
        if 'instances' in info and len(info['instances']) > 0:
            ann_info = dict(
                gt_bboxes_3d=np.zeros((len(info['instances']), 9), dtype=np.float32),
                gt_labels_3d=np.zeros((len(info['instances']),), dtype=np.int64),
            )
            for idx, instance in enumerate(info['instances']):
                ann_info['gt_bboxes_3d'][idx] = instance['bbox_3d']
                ann_info['gt_labels_3d'][idx] = self.label_mapping[instance['bbox_label_3d']]

        if ann_info is None:
            ann_info = dict(
                gt_bboxes_3d=np.zeros((0, 9), dtype=np.float32),
                gt_labels_3d=np.zeros((0,), dtype=np.int64),
            )

        # Optional visible instance masks
        if 'visible_instance_ids' in info['images'][0]:
            ids = [im['visible_instance_ids'] for im in info['images']]
            mask_length = ann_info['gt_labels_3d'].shape[0]
            ann_info['visible_instance_masks'] = self._ids2masks(ids, mask_length)

        # Remove 'dontcare' labels (-1) from box labels, keep occupancy intact
        ann_info = self._remove_dontcare(ann_info)

        # === Occupancy (point list) ===
        occ_filename = os.path.join(
            self.data_prefix.get('img_path', ''), 'occupancy_2x', info['sample_idx'], 'occupancy.npy')
        gt_occ = np.load(occ_filename).astype(np.int64)  # shape (N, 4): [x, y, z, label]
        ann_info['gt_occupancy'] = gt_occ

        # Visible occupancy mask (unused here; keep interface)
        ann_info['visible_occupancy_masks'] = None
        return ann_info

    def parse_data_info(self, info: dict) -> dict:
        """Normalize paths/matrices and attach ann_info for training/val."""
        info['axis_align_matrix'] = self._get_axis_align_matrix(info)
        info['scan_id'] = info['sample_idx']

        # Depth scale default
        ann_dataset = info['sample_idx'].split('/')[0]
        info['depth_shift'] = 4000.0 if ann_dataset == 'matterport3d' else 1000.0

        # Collect per-view image/depth paths and extrinsics
        info['img_path'] = []
        info['depth_img_path'] = []

        if 'cam2img' in info:
            cam2img = info['cam2img'].astype(np.float32)
        else:
            cam2img = []

        extrinsics = []
        for i in range(len(info['images'])):
            img_path = os.path.join(self.data_prefix.get('img_path', ''), info['images'][i]['img_path'])
            depth_img_path = os.path.join(self.data_prefix.get('img_path', ''), info['images'][i]['depth_path'])
            info['img_path'].append(img_path)
            info['depth_img_path'].append(depth_img_path)

            align_global2cam = np.linalg.inv(info['axis_align_matrix'] @ info['images'][i]['cam2global'])
            extrinsics.append(align_global2cam.astype(np.float32))

            if 'cam2img' not in info:
                cam2img.append(info['images'][i]['cam2img'].astype(np.float32))

        occ_filename = os.path.join(self.data_prefix.get('img_path', ''),
                                    'occupancy_2x', info['sample_idx'], 'occupancy.npy')
        mask_filename = os.path.join(self.data_prefix.get('img_path', ''),
                                    'occupancy_2x', info['sample_idx'], 'visible_occupancy.pkl')
        info['occ_path'] = occ_filename
        info['occ_mask_path'] = mask_filename

        info['depth2img'] = dict(
            extrinsic=extrinsics,
            intrinsic=cam2img,
            origin=np.array([.0, .0, .0]).astype(np.float32)
        )

        if 'depth_cam2img' not in info:
            info['depth_cam2img'] = cam2img

        # Attach ann_info for train/val; skip for pure test if needed
        if self.phase != 'test':
            info['ann_info'] = self.parse_ann_info(info)
            if 'gt_occupancy' in info['ann_info'] and info['ann_info']['gt_occupancy'].shape[0] == 0:
                return None

        return info

    def load_data_list(self) -> List[dict]:
        """Load ann_file → merge metainfo → build data_list via parse_data_info()."""
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'annotations should be a dict, but got {type(annotations)}')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('ann_file must contain keys: data_list, metainfo')

        # Merge metainfo and build mappings
        for k, v in annotations['metainfo'].items():
            self._metainfo.setdefault(k, v)
        self.process_metainfo()

        # Parse each raw record
        data_list = []
        for raw in annotations['data_list']:
            data_info = self.parse_data_info(raw)
            if isinstance(data_info, dict):
                data_list.append(data_info)
        return data_list

    @staticmethod
    def _ids2masks(ids, mask_length):
        """Convert per-view visible instance ids to boolean masks."""
        masks = []
        for idx in range(len(ids)):
            mask = np.zeros((mask_length,), dtype=bool)
            mask[ids[idx]] = 1
            masks.append(mask)
        return masks

    def _remove_dontcare(self, ann_info: dict) -> dict:
        """Filter out instances with label -1; keep occupancy entries as-is."""
        out = {}
        if 'gt_labels_3d' in ann_info:
            filter_mask = ann_info['gt_labels_3d'] > -1
        else:
            filter_mask = None

        for key in ann_info.keys():
            if key in ['gt_occupancy', 'visible_occupancy_masks']:
                out[key] = ann_info[key]
            elif key == 'visible_instance_masks' and filter_mask is not None:
                out[key] = [m[filter_mask] for m in ann_info[key]]
            elif key in ['gt_bboxes_3d', 'gt_labels_3d'] and filter_mask is not None:
                out[key] = ann_info[key][filter_mask]
            else:
                out[key] = ann_info[key]
        return out

    # ---------------------------
    # PyTorch Dataset interface
    # ---------------------------
    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        """Return (imgs, meta, occs) for the given index.

        imgs: (1, N_views, H, W, 3) float32 RGB, resized to `resize_hw`
        meta: dict with intrinsics/extrinsics (resized), paths, and misc
        occs: (N_pts, 4) int64 [x, y, z, label] from parse_ann_info()
        """
        info = self.data_list[idx]

        meta = {}
        meta['name'] = info['scan_id']
        meta['scene_size'] = self.scene_size
        num_scene_images = len(info['img_path'])
        replace = num_scene_images < self.num_frames
        sel_idx = np.random.choice(num_scene_images, self.num_frames, replace=replace)

        # Original code assumes self.num_frames == 1
        sel_idx = sel_idx[0]

        # gather img_path, depth_img_path, cam2img, depth2img.extrinsics
        # rgb_path = [info['img_path'][i] for i in sel_idx]
        # depth_path = [info['depth_img_path'][i] for i in sel_idx]
        # world2cam = [info['depth2img']['extrinsic'][i] for i in sel_idx]
        # org_cam_intrinsic = info['depth2img']['intrinsic']
        # world2cam = np.stack(world2cam, axis=0)  # (N_views, 4, 4)
        # cam2world = np.linalg.inv(world2cam)  # (N_views, 4, 4)
        # meta['rgb_path'] = rgb_path
        # meta['cam2world'] = cam2world
        # meta['world2cam'] = world2cam

        rgb_path = info['img_path'][sel_idx]
        depth_path = info['depth_img_path'][sel_idx]
        world2cam = info['depth2img']['extrinsic'][sel_idx]
        org_cam_intrinsic = info['depth2img']['intrinsic']
        cam2world = np.linalg.inv(world2cam)
        meta['rgb_path'] = rgb_path
        meta['cam2world'] = cam2world
        meta['world2cam'] = world2cam

        depth_gt_np = Image.open(depth_path).convert('I;16')
        depth_gt_np = np.array(depth_gt_np) / 1000.0

        transform = Compose([
            Resize(
                width=480,
                height=480,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        img_depthbranch = cv2.imread(rgb_path)
        img_depthbranch = cv2.resize(img_depthbranch, (640, 480), interpolation=cv2.INTER_NEAREST)
        img_depthbranch = cv2.cvtColor(img_depthbranch, cv2.COLOR_BGR2RGB) / 255.0
        sample = transform({'image': img_depthbranch, 'depth': depth_gt_np})
        img_depthbranch = torch.from_numpy(sample['image']).unsqueeze(0)
        depth_gt_np = torch.from_numpy(sample['depth']).unsqueeze(0)
        meta['depth_gt_np'] = depth_gt_np
        depth_valid_mask = (torch.isnan(depth_gt_np) == 0)
        depth_gt_np[depth_valid_mask == 0] = 0
        meta['img_depthbranch'] = img_depthbranch
        meta['depth_gt_np_valid'] = depth_gt_np

        meta['rgb_path'] = rgb_path
        N_img = []
        this_img = imread(rgb_path, 'unchanged').astype(np.float32)
        this_H, this_W, _ = this_img.shape
        new_H, new_W = 480, 640
        # resize
        new_img = cv2.resize(this_img, (new_W, new_H))
        W_factor = new_W / this_W
        H_factor = new_H / this_H
        N_img.append(new_img)
        img = np.stack(N_img, 0) # [1, 968, 1296, 3]
        this_H, this_W= new_H, new_W
        img = [img] # [1, 1, 968, 1296, 3]
        
        cam_intrin = org_cam_intrinsic.copy()
        cam_intrin[0, 0] *= W_factor
        cam_intrin[0, 2] *= W_factor
        cam_intrin[1, 1] *= H_factor
        cam_intrin[1, 2] *= H_factor
        
        meta['cam_k'] = cam_intrin[:3, :3]
        viewpad = np.eye(4)
        viewpad[:meta['cam_k'].shape[0], :meta['cam_k'].shape[1]] = meta['cam_k']
        meta['cam2img'] = viewpad
        world2img = (viewpad @ world2cam)
        meta['world2img'] = world2img

        meta['depth_path'] = depth_path
        depth_gt = Image.open(depth_path).convert('I;16')
        depth_gt = np.array(depth_gt) / 1000.0
        meta['depth_gt'] = depth_gt

        vox_origin = self.scene_min_bound
        meta['vox_origin'] = np.round(np.array(vox_origin, dtype=np.float32), 4)
        target = np.load(info['occ_path']).astype(np.int64)  # shape (N, 4): [x, y, z, label]
        target[target == 0] = 12
        target[target == 255] = 0 

        vox_W, vox_H, vox_D = self.grid_size_occ
        # target to dense occ
        occ = np.zeros((vox_W, vox_H, vox_D), dtype=np.uint8)
        occ[target[:, 0], target[:, 1], target[:, 2]] = target[:, 3].astype(np.uint8)
        nonemptymask = (occ != 12)
        occ = [occ] # [1, 240, 240, 80]
        
        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z, occ_xyz = vox2pix(
            world2cam,
            meta['cam_k'],
            meta['vox_origin'],
            self.voxel_size,
            this_W,
            this_H,
            self.scene_size,
            dim_60_60_36=True,
            voxel_dims=(vox_W, vox_H, vox_D)
        )
        _, fov_mask_4, _, _ = vox2pix(
            world2cam,
            meta['cam_k'],
            meta['vox_origin'],
            self.voxel_size * 4,
            this_W,
            this_H,
            self.scene_size,
            dim_60_60_36=False,
            voxel_dims=(vox_W, vox_H, vox_D)
        )

        meta['projected_pix'] = projected_pix
        meta['fov_mask'] = fov_mask.reshape(vox_W, vox_H, vox_D)
        meta['fov_mask_4'] = fov_mask_4.reshape(vox_W // 4, vox_H // 4, vox_D // 4)
        
        meta['pix_z'] = pix_z
        meta['occ_xyz'] = occ_xyz.reshape(vox_W, vox_H, vox_D, 3)
        
        vox_near = meta['vox_origin']
        vox_far = vox_near + meta['scene_size']
        nyu_pc_range = np.concatenate([vox_near, vox_far], axis=0)
        meta['nyu_pc_range'] = nyu_pc_range
        
        scan = meta['occ_xyz'][nonemptymask]
        meta['occ_xyz_nonempty'] = scan
        meta['num_depth'] = self.num_pts
        if scan.shape[0] < self.num_pts:
            multi = int(math.ceil(self.num_pts * 1.0 / scan.shape[0])) - 1
            scan_ = np.repeat(scan, multi, 0)
            scan_ = scan_ + np.random.randn(*scan_.shape) * 0.01
            scan_ = scan_[np.random.choice(scan_.shape[0], self.num_pts - scan.shape[0], False)]
            scan_[:, 0] = np.clip(scan_[:, 0], nyu_pc_range[0], nyu_pc_range[3])
            scan_[:, 1] = np.clip(scan_[:, 1], nyu_pc_range[1], nyu_pc_range[4])
            scan_[:, 2] = np.clip(scan_[:, 2], nyu_pc_range[2], nyu_pc_range[5])
            scan = np.concatenate([scan, scan_], 0)
        else:
            scan = scan[np.random.choice(scan.shape[0], self.num_pts, False)]

        # import open3d as o3d
        # # fov_mask = fov_mask.reshape(240, 240, 80)
        # # idxs = np.where(fov_mask)
        # # idxs = np.vstack(idxs).T.astype(np.float32)
        # o3d_cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
        # o3d_cam_axis.transform(cam2world)
        # o3d_pcd = o3d.geometry.PointCloud()
        # #o3d_pcd.points = o3d.utility.Vector3dVector((idxs + 0.5) * self.voxel_size + np.array(vox_origin)[None, :])
        # o3d_pcd.points = o3d.utility.Vector3dVector(occ_xyz[fov_mask, :])
        # o3d_occ = o3d.geometry.PointCloud()
        # o3d_occ.points = o3d.utility.Vector3dVector(target[:, :3] * self.voxel_size + np.array(vox_origin)[None, :])
        # o3d_occ.paint_uniform_color([1.0, 0.0, 0.0])
        # o3d_pcd_scan = o3d.geometry.PointCloud()
        # o3d_pcd_scan.points = o3d.utility.Vector3dVector(scan[:, :3])
        # o3d.visualization.draw_geometries([o3d_pcd_scan, o3d_cam_axis, o3d_occ])
        
        scan[:, 0] = (scan[:, 0] - nyu_pc_range[0]) / (nyu_pc_range[3] - nyu_pc_range[0])
        scan[:, 1] = (scan[:, 1] - nyu_pc_range[1]) / (nyu_pc_range[4] - nyu_pc_range[1])
        scan[:, 2] = (scan[:, 2] - nyu_pc_range[2]) / (nyu_pc_range[5] - nyu_pc_range[2])
        
        meta['anchor_points'] = scan
        
        cam_vox_near = np.array([-5, -6, -3])
        cam_vox_far = np.array([5, 6, 8])
        cam_vox_range = np.concatenate([cam_vox_near, cam_vox_far], axis=0).astype(np.float32)
        meta['cam_vox_range'] = cam_vox_range
        
        meta['occ_mask_valid'] = (occ != 0)
        meta['occ_mask_valid_fov'] = (occ != 0) & fov_mask
        meta['label'] = occ
        imgs = np.stack(img, 0)
        occs = np.stack(occ, 0)
        data_tuple = (imgs, meta, occs)
        return data_tuple
