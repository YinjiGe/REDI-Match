import os
from PIL import Image
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as tvf
import kornia.augmentation as K
import cv2
from redimatch.utils import get_depth_tuple_transform_ops, get_tuple_transform_ops
import redimatch
from redimatch.utils import *
import math


def _k_mod4_to_angle_deg(k_mod4: int) -> int:
    return {0: 0, 1: 90, 2: 180, 3: 270}[int(k_mod4) % 4]


def _rotate_rgb_native(arr: np.ndarray, angle_deg: int) -> np.ndarray:
    if angle_deg == 0:
        return np.asarray(arr).copy()
    if angle_deg == 90:
        return cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle_deg == 180:
        return cv2.rotate(arr, cv2.ROTATE_180)
    if angle_deg == 270:
        return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(angle_deg)


def _rotate_depth_native(depth: np.ndarray, angle_deg: int) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    return _rotate_rgb_native(d, angle_deg)


def _rotate_intrinsics_native(K, w: int, h: int, angle_deg: int):
    if angle_deg == 0:
        return K
    Kn = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    fx, fy = Kn[0, 0], Kn[1, 1]
    cx, cy = Kn[0, 2], Kn[1, 2]
    if angle_deg == 90:
        Kn[0, 0], Kn[1, 1] = fy, fx
        Kn[0, 2] = cy
        Kn[1, 2] = w - 1 - cx
    elif angle_deg == 180:
        Kn[0, 2] = w - 1 - cx
        Kn[1, 2] = h - 1 - cy
    elif angle_deg == 270:
        Kn[0, 0], Kn[1, 1] = fy, fx
        Kn[0, 2] = h - 1 - cy
        Kn[1, 2] = cx
    else:
        raise ValueError(angle_deg)
    if torch.is_tensor(K):
        return torch.tensor(Kn, dtype=K.dtype, device=K.device)
    return Kn

class MegadepthScene:
    def __init__(
        self,
        data_root,
        scene_info,
        ht=384,
        wt=512,
        min_overlap=0.0,
        max_overlap=1.0,
        shake_t=0,
        rot_prob=0.0,
        normalize=True,
        max_num_pairs = 100_000,
        scene_name = None,
        use_horizontal_flip_aug = False,
        use_single_horizontal_flip_aug = False,
        colorjiggle_params = None,
        random_eraser = None,
        use_randaug = False,
        randaug_params = None,
        randomize_size = False,
        resize_mode: str = "stretch",
        rot_b_tensor_equivariant: bool = False,
        rotation_gt_map: dict | None = None,
        plain_asset_root: str | None = None,
        b_side: str = "brot",
    ) -> None:
        self.data_root = data_root
        self.scene_name = os.path.splitext(scene_name)[0]+f"_{min_overlap}_{max_overlap}"
        self.image_paths = scene_info["image_paths"]
        self.image_paths_b_rot = scene_info.get("image_paths_b_rot")
        self.depth_paths = scene_info["depth_paths"]
        self.depth_paths_b_rot = scene_info.get("depth_paths_b_rot")
        self.intrinsics = scene_info["intrinsics"]
        self.intrinsics_b_rot = scene_info.get("intrinsics_b_rot")
        self.poses = scene_info["poses"]
        self.poses_b_rot = scene_info.get("poses_b_rot")
        fa = scene_info.get("fixed_a_only_rotate_b")
        if fa is None:
            self.fixed_a_only_rotate_b = False
        elif isinstance(fa, np.ndarray):
            self.fixed_a_only_rotate_b = bool(fa.item() if fa.size == 1 else fa.any())
        else:
            self.fixed_a_only_rotate_b = bool(fa)
        self.pairs = scene_info["pairs"]
        self.overlaps = scene_info["overlaps"]
        threshold = (self.overlaps > min_overlap) & (self.overlaps < max_overlap)
        self.pairs = self.pairs[threshold]
        self.overlaps = self.overlaps[threshold]
        if len(self.pairs) > max_num_pairs:
            pairinds = np.random.choice(
                np.arange(0, len(self.pairs)), max_num_pairs, replace=False
            )
            self.pairs = self.pairs[pairinds]
            self.overlaps = self.overlaps[pairinds]
        if randomize_size:
            area = ht * wt
            s = int(16 * (math.sqrt(area)//16))
            sizes = ((ht,wt), (s,s), (wt,ht))
            choice = redimatch.RANK % 3
            ht, wt = sizes[choice] 
        # counts, bins = np.histogram(self.overlaps,20)
        # print(counts)
        self.im_transform_ops = get_tuple_transform_ops(
            resize=(ht, wt),
            normalize=normalize,
            colorjiggle_params=colorjiggle_params,
            resize_mode=str(resize_mode),
        )
        self.depth_transform_ops = get_depth_tuple_transform_ops(
            resize=(ht, wt), resize_mode=str(resize_mode)
        )
        self.resize_mode = str(resize_mode).lower()
        self.rot_b_tensor_equivariant = bool(rot_b_tensor_equivariant)
        self.rotation_gt_map = rotation_gt_map or {}
        self.plain_asset_root = (
            os.path.abspath(plain_asset_root) if plain_asset_root else self.data_root
        )
        self.b_side = str(b_side).lower()
        self.wt, self.ht = wt, ht
        self.shake_t = shake_t
        self.random_eraser = random_eraser
        if use_horizontal_flip_aug and use_single_horizontal_flip_aug:
            raise ValueError("Can't both flip both images and only flip one")
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.use_single_horizontal_flip_aug = use_single_horizontal_flip_aug
        self.use_randaug = use_randaug

    def load_im(self, im_path):
        im = Image.open(im_path)
        return im
    
    def horizontal_flip(self, im_A, im_B, depth_A, depth_B,  K_A, K_B):
        im_A = im_A.flip(-1)
        im_B = im_B.flip(-1)
        depth_A, depth_B = depth_A.flip(-1), depth_B.flip(-1) 
        flip_mat = torch.tensor([[-1, 0, self.wt],[0,1,0],[0,0,1.]]).to(K_A.device)
        K_A = flip_mat@K_A  
        K_B = flip_mat@K_B  
        
        return im_A, im_B, depth_A, depth_B, K_A, K_B
    
    def load_depth(self, depth_ref, crop=None):
        depth = np.array(h5py.File(depth_ref, "r")["depth"])
        return torch.from_numpy(depth)

    def __len__(self):
        return len(self.pairs)

    def scale_intrinsic(self, K, wi, hi):
        if self.resize_mode == "letterbox":
            sx = sy = self.wt / float(max(wi, hi))
            nw = max(1, int(round(wi * sx)))
            nh = max(1, int(round(hi * sy)))
            ox = (self.wt - nw) // 2
            oy = (self.ht - nh) // 2
            K = K.clone()
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] = K[0, 2] * sx + float(ox)
            K[1, 2] = K[1, 2] * sy + float(oy)
            return K
        sx, sy = self.wt / wi, self.ht / hi
        sK = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return sK @ K

    def _rot_b_tensor_k(self, rel_brot: str) -> int:
        rel = str(rel_brot).replace("\\", "/")
        if rel.startswith("Undistorted_SfM/"):
            rel = rel.split("Undistorted_SfM/", 1)[1]
        k = self.rotation_gt_map.get(rel)
        if k is None:
            return 0
        return int(k) % 4

    def _apply_native_rot_b(self, im_B, depth_B, K2, k_mod4: int):
        """Consistent with prep: rotate B at original resolution, then stretch/letterbox (not rot90(stretch))."""
        angle = _k_mod4_to_angle_deg(k_mod4)
        if angle == 0:
            return im_B, depth_B, K2
        wi, hi = im_B.size
        rgb = _rotate_rgb_native(np.array(im_B), angle)
        im_B = Image.fromarray(rgb)
        depth_B = torch.from_numpy(_rotate_depth_native(depth_B.numpy(), angle))
        K2 = _rotate_intrinsics_native(K2, wi, hi, angle)
        return im_B, depth_B, K2

    def _apply_tensor_rot_b(self, im_B, depth_B, K2, k_mod4: int):
        k = int(k_mod4) % 4
        if k == 0:
            return im_B, depth_B, K2
        im_B = torch.rot90(im_B, k=k, dims=[-2, -1])
        depth_B = torch.rot90(depth_B, k=k, dims=[-2, -1])
        w, h = int(self.wt), int(self.ht)
        Kn = K2.clone().double()
        fx, fy, cx, cy = Kn[0, 0], Kn[1, 1], Kn[0, 2], Kn[1, 2]
        if k == 1:
            Kn[0, 0], Kn[1, 1] = fy, fx
            Kn[0, 2] = cy
            Kn[1, 2] = w - 1 - cx
        elif k == 2:
            Kn[0, 2] = w - 1 - cx
            Kn[1, 2] = h - 1 - cy
        elif k == 3:
            Kn[0, 0], Kn[1, 1] = fy, fx
            Kn[0, 2] = h - 1 - cy
            Kn[1, 2] = cx
        return im_B, depth_B, Kn.float()

    def rand_shake(self, *things):
        t = np.random.choice(range(-self.shake_t, self.shake_t + 1), size=2)
        return [
            tvf.affine(thing, angle=0.0, translate=list(t), scale=1.0, shear=[0.0, 0.0])
            for thing in things
        ], t

    def __getitem__(self, pair_idx):
        # read intrinsics of original size
        idx1, idx2 = self.pairs[pair_idx]
        K1 = torch.tensor(self.intrinsics[idx1].copy(), dtype=torch.float).reshape(3, 3)
        K2 = torch.tensor(self.intrinsics[idx2].copy(), dtype=torch.float).reshape(3, 3)
        T1 = self.poses[idx1]

        # Load positive pair data（megadepth_rot_2：A=plain，B=Brot）
        im_A = self.image_paths[idx1]
        use_plain_b = (
            self.fixed_a_only_rotate_b
            and self.b_side == "plain"
            and self.image_paths[idx2] is not None
        )
        if use_plain_b:
            im_B = self.image_paths[idx2]
            depth2 = self.depth_paths[idx2]
            K2 = torch.tensor(self.intrinsics[idx2].copy(), dtype=torch.float).reshape(3, 3)
            T2 = self.poses[idx2]
            im_B_id = im_B.split("/")[-1].split(".jpg")[0]
        elif self.fixed_a_only_rotate_b and self.image_paths_b_rot is not None:
            im_B = self.image_paths_b_rot[idx2]
            if im_B is None:
                im_B = self.image_paths[idx2]
            depth2 = (
                self.depth_paths_b_rot[idx2]
                if self.depth_paths_b_rot is not None
                else self.depth_paths[idx2]
            )
            if self.intrinsics_b_rot is not None and self.intrinsics_b_rot[idx2] is not None:
                K2 = torch.tensor(
                    self.intrinsics_b_rot[idx2].copy(), dtype=torch.float
                ).reshape(3, 3)
            T2 = (
                self.poses_b_rot[idx2]
                if self.poses_b_rot is not None and self.poses_b_rot[idx2] is not None
                else self.poses[idx2]
            )
            im_B_id = im_B.split("/")[-1].split(".jpg")[0]
        else:
            im_B = self.image_paths[idx2]
            depth2 = self.depth_paths[idx2]
            T2 = self.poses[idx2]
            im_B_id = im_B.split("/")[-1].split(".jpg")[0]
        depth1 = self.depth_paths[idx1]
        T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1)), dtype=torch.float)[:4, :4]
        im_A_ref = os.path.join(self.data_root, im_A)
        im_B_ref = os.path.join(self.data_root, im_B)
        depth_A_ref = os.path.join(self.data_root, depth1)
        depth_B_ref = os.path.join(self.data_root, depth2)
        use_tensor_rot = (
            self.rot_b_tensor_equivariant
            and self.fixed_a_only_rotate_b
            and self.image_paths_b_rot is not None
            and self.b_side != "plain"
        )
        if use_tensor_rot:
            plain_b = self.image_paths[idx2]
            if plain_b is None and im_B is not None:
                plain_b = str(im_B).replace("_Brot.jpg", ".jpg").replace("_brot.jpg", ".jpg")
            if plain_b is None:
                use_tensor_rot = False
            elif self.intrinsics[idx2] is None:
                use_tensor_rot = False
            else:
                im_B_ref = os.path.join(self.plain_asset_root, plain_b)
                plain_dep = self.depth_paths[idx2]
                if plain_dep is None and depth2 is not None:
                    plain_dep = str(depth2).replace("_Brot", "").replace("_brot", "")
                depth_B_ref = os.path.join(self.plain_asset_root, plain_dep)
                K2 = torch.tensor(self.intrinsics[idx2].copy(), dtype=torch.float).reshape(3, 3)
                rel_brot_for_k = self.image_paths_b_rot[idx2]
        im_A = self.load_im(im_A_ref)
        im_B = self.load_im(im_B_ref)
        depth_A = self.load_depth(depth_A_ref)
        depth_B = self.load_depth(depth_B_ref)

        tensor_rot_post = False
        k_b = 0
        if use_tensor_rot:
            k_b = self._rot_b_tensor_k(rel_brot_for_k)
            if self.resize_mode == "letterbox":
                tensor_rot_post = True
            else:
                im_B, depth_B, K2 = self._apply_native_rot_b(im_B, depth_B, K2, k_b)

        K1 = self.scale_intrinsic(K1, im_A.width, im_A.height)
        K2 = self.scale_intrinsic(K2, im_B.width, im_B.height)

        if self.use_randaug:
            im_A, im_B = self.rand_augment(im_A, im_B)

        # Process images
        im_A, im_B = self.im_transform_ops((im_A, im_B))
        depth_A, depth_B = self.depth_transform_ops(
            (depth_A[None, None], depth_B[None, None])
        )
        if tensor_rot_post:
            im_B, depth_B, K2 = self._apply_tensor_rot_b(im_B, depth_B, K2, k_b)
        
        [im_A, im_B, depth_A, depth_B], t = self.rand_shake(im_A, im_B, depth_A, depth_B)
        K1[:2, 2] += t
        K2[:2, 2] += t
        
        im_A, im_B = im_A[None], im_B[None]
        if self.random_eraser is not None:
            im_A, depth_A = self.random_eraser(im_A, depth_A)
            im_B, depth_B = self.random_eraser(im_B, depth_B)
                
        if self.use_horizontal_flip_aug:
            if np.random.rand() > 0.5:
                im_A, im_B, depth_A, depth_B, K1, K2 = self.horizontal_flip(im_A, im_B, depth_A, depth_B, K1, K2)
        if self.use_single_horizontal_flip_aug:
            if np.random.rand() > 0.5:
                im_B, depth_B, K2 = self.single_horizontal_flip(im_B, depth_B, K2)
        
        if redimatch.DEBUG_MODE:
            tensor_to_pil(im_A[0], unnormalize=True).save(
                            f"vis/im_A.jpg")
            tensor_to_pil(im_B[0], unnormalize=True).save(
                            f"vis/im_B.jpg")
            
        data_dict = {
            "im_A": im_A[0],
            "im_A_identifier": self.image_paths[idx1].split("/")[-1].split(".jpg")[0],
            "im_B": im_B[0],
            "im_B_identifier": im_B_id,
            "im_A_depth": depth_A[0, 0],
            "im_B_depth": depth_B[0, 0],
            "K1": K1,
            "K2": K2,
            "T_1to2": T_1to2,
            "im_A_path": im_A_ref,
            "im_B_path": im_B_ref,
        }
        if self.fixed_a_only_rotate_b and self.image_paths_b_rot is not None:
            brot_rel = self.image_paths_b_rot[idx2]
            if brot_rel is not None:
                data_dict["im_B_rot_path"] = os.path.join(self.data_root, brot_rel)
        return data_dict


class MegadepthBuilder:
    def __init__(
        self,
        data_root="data/megadepth",
        loftr_ignore=True,
        imc21_ignore=True,
        scene_info_root=None,
    ) -> None:
        self.data_root = data_root
        self.scene_info_root = (
            os.path.abspath(scene_info_root)
            if scene_info_root is not None
            else os.path.join(data_root, "prep_scene_info")
        )
        self.all_scenes = os.listdir(self.scene_info_root)
        self.test_scenes = ["0017.npy", "0004.npy", "0048.npy", "0013.npy"]
        # LoFTR did the D2-net preprocessing differently than we did and got more ignore scenes, can optionially ignore those
        self.loftr_ignore_scenes = set(['0121.npy', '0133.npy', '0168.npy', '0178.npy', '0229.npy', '0349.npy', '0412.npy', '0430.npy', '0443.npy', '1001.npy', '5014.npy', '5015.npy', '5016.npy'])
        self.imc21_scenes = set(['0008.npy', '0019.npy', '0021.npy', '0024.npy', '0025.npy', '0032.npy', '0063.npy', '1589.npy'])
        self.test_scenes_loftr = ["0015.npy", "0022.npy"]
        self.loftr_ignore = loftr_ignore
        self.imc21_ignore = imc21_ignore

    def build_scenes(self, split="test", min_overlap=0.0, scene_names = None, **kwargs):
        if split == "test":
            scene_names = self.test_scenes
        elif split == "test_loftr":
            scene_names = self.test_scenes_loftr
        elif split == "custom":
            scene_names = scene_names
        else:
            raise ValueError(
                f"Public evaluation runtime supports only test/test_loftr/custom, got {split!r}"
            )
        scenes = []
        for scene_name in scene_names:
            if self.loftr_ignore and scene_name in self.loftr_ignore_scenes:
                continue
            if self.imc21_ignore and scene_name in self.imc21_scenes:
                continue
            if ".npy" not in scene_name:
                continue
            scene_info = np.load(
                os.path.join(self.scene_info_root, scene_name), allow_pickle=True
            ).item()
            scenes.append(
                MegadepthScene(
                    self.data_root, scene_info, min_overlap=min_overlap,scene_name = scene_name, **kwargs
                )
            )
        return scenes

    def weight_scenes(self, concat_dataset, alpha=0.5):
        ns = []
        for d in concat_dataset.datasets:
            ns.append(len(d))
        ws = torch.cat([torch.ones(n) / n**alpha for n in ns])
        return ws
