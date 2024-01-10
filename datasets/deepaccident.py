import json
import logging
import os
from typing import Dict

import numpy as np
import torch
from nuscenes.nuscenes import LidarPointCloud, NuScenes
from omegaconf import OmegaConf
from pyquaternion import Quaternion
from torch import Tensor
from tqdm import trange
import pyquaternion
from datasets.base.lidar_source import SceneLidarSource
from datasets.base.pixel_source import ScenePixelSource
from datasets.base.scene_dataset import SceneDataset
from datasets.base.split_wrapper import SplitWrapper
from datasets.utils import voxel_coords_to_world_coords
from radiance_fields.video_utils import save_videos, depth_visualizer
from utils.misc import NumpyEncoder
import pickle
logger = logging.getLogger()
class DeepAccidentPixelSource(ScenePixelSource):
    ORIGINAL_SIZE = [[900, 1600] for _ in range(6)]
    OPENCV2DATASET = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def __init__(
        self,
        pixel_data_config: OmegaConf,
        data_path: str,
        meta_file_path: str,
        nusc: NuScenes = None,
        scene_idx: int = 0,
        start_timestep: int = 0,
        end_timestep: int = -1,
        device: torch.device = torch.device("cpu"),
    ):
        pixel_data_config.load_dynamic_mask = False
        logger.info("[Pixel] Overriding load_dynamic_mask to False")
        super().__init__(pixel_data_config, device=device)

        self.data_path = data_path
        self.meta_file_path = meta_file_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.nusc = nusc
        self.scene_idx = scene_idx
        self.meta_dict = self.create_or_load_metas()
        self.create_all_filelist()

        self.load_data()


    def create_or_load_metas(self):
        # ---- define camera list ---- #
        # ---- define camera list ---- #
        #if self.num_cams == 1:
         #   self.camera_list = ["CAM_FRONT"]
        #elif self.num_cams == 3:
         #   self.camera_list = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
        if self.num_cams == 6:
            self.camera_list = [
                "Camera_FrontLeft",
                "Camera_Front",
                "Camera_FrontRight",
                "Camera_BackLeft",
                "Camera_Back",
                "Camera_BackRight",
            ]
        else:
            raise NotImplementedError(
                f"num_cams: {self.num_cams} not supported for nuscenes dataset"
            )


        data_raw = pickle.load(open(self.meta_file_path, 'rb'))

        if os.path.exists(self.meta_file_path):
            with open("/home/uchihadj/EmerNeRF/sample.json", "r") as f:
                meta_dict = json.load(f)
            logger.info(f"[Pixel] Loaded camera meta from {self.meta_file_path}")
            return meta_dict

        self.data_info_all = {}
        data_infos = []
        for data in data_raw['infos']:
            key_name = data['scene_name'] + '_' + data['vehicle_name'] + '_' + str(data['timestamp'])
            self.data_info_all[key_name] = data

            if ((data['scenario_length'] - data['timestamp']) <= 4 * 5) and (
                    (data['scenario_length'] - data['timestamp']) > 0) \
                    and 'accident' in data['scene_name']:
                data_infos.append(data)
                data_infos = list(
                sorted(data_infos, key=lambda x: (x['scene_name'], x['vehicle_name'], x['timestamp']), reverse=False))
                data_infos = data_infos[:50]


        # Initialize the meta_dict
        meta_dict = {
            "Camera_FrontLeft": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
                               "intrinsics": []},
            "Camera_Front": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
                          "intrinsics": []},
            "Camera_FrontRight": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
                                "intrinsics": []},
            "Camera_BackLeft": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
                              "intrinsics": []},
            "Camera_Back": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
                         "intrinsics": []},
            "Camera_BackRight": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
                               "intrinsics": []}
        }
        #meta_dict["CAMERA_FRONTLEFT"]['timestamp'].append(img_infos["Camera_FrontLe"])


        # Map camera names to IDs
        camera_id_mapping = {
            "Camera_FrontLeft": 0,
            "Camera_Front": 1,
            "Camera_FrontRight": 2,
            "Camera_BackLeft": 3,
            "Camera_Back": 4,
            "Camera_BackRight": 5
        }
        # ---- find the minimum shared scene length ---- #
        #info = self.data_infos

        for i in range(len(data_infos)):

            info = data_infos[i]

            print("just one info", info)
            img_infos = []
            img_infos.append(info['cams'])

            print(camera_id_mapping['Camera_FrontLeft'])
            ########################################################################
            """Parse all the camera detials to meta_dict"""
            ########################################################################
            meta_dict["Camera_FrontLeft"]['timestamp'].append(img_infos[0]["Camera_FrontLeft"]["timestamp"])
            meta_dict["Camera_FrontLeft"]["filepath"].append(img_infos[0]["Camera_FrontLeft"]["image_path"])
            meta_dict["Camera_FrontLeft"]['ego_pose'].append(info['ego_to_world_matrix'])
            meta_dict["Camera_FrontLeft"]['cam_id'].append(camera_id_mapping['Camera_FrontLeft'])
            meta_dict["Camera_FrontLeft"]['intrinsics'].append(
                img_infos[0]["Camera_FrontLeft"]["camera_intrinsic_matrix"])
            # check for camera visisblity is
            # meta_dict["Camera_FrontLef"]['kind_of_cam_id'].append(info['camera_visibility'])
            #compute extrinsics
            e2g_trans_matrix = np.zeros((4, 4), dtype=np.float32)
            e2g_rot = info['ego2global_rotation']
            e2g_trans = info['ego2global_translation']
            e2g_trans_matrix[:3, :3] = pyquaternion.Quaternion(
                e2g_rot).rotation_matrix
            e2g_trans_matrix[:3, 3] = np.array(e2g_trans)
            e2g_trans_matrix[3, 3] = 1.0
            print(e2g_trans_matrix)
            #old_function for extrinsics
            meta_dict["Camera_FrontLeft"]['extrinsics'].append(info["lidar_to_ego_matrix"])
            #meta_dict["Camera_FrontLeft"]['extrinsics'].append(e2g_trans_matrix)

            # second camera
            meta_dict["Camera_Front"]['timestamp'].append(img_infos[0]["Camera_Front"]["timestamp"])
            meta_dict["Camera_Front"]["filepath"].append(img_infos[0]["Camera_Front"]["image_path"])
            meta_dict["Camera_Front"]['ego_pose'].append(info['ego_to_world_matrix'])
            meta_dict["Camera_Front"]['cam_id'].append(camera_id_mapping['Camera_Front'])
            meta_dict["Camera_Front"]['intrinsics'].append(img_infos[0]["Camera_Front"]["camera_intrinsic_matrix"])
            # check for camera visisblity is
            # meta_dict["Camera_FrontLef"]['kind_of_cam_id'].append(info['camera_visibility'])
            meta_dict["Camera_Front"]['extrinsics'].append(info["lidar_to_ego_matrix"])
            # third camera
            meta_dict["Camera_FrontRight"]['timestamp'].append(img_infos[0]["Camera_FrontRight"]["timestamp"])
            meta_dict["Camera_FrontRight"]["filepath"].append(img_infos[0]["Camera_FrontRight"]["image_path"])
            meta_dict["Camera_FrontRight"]['ego_pose'].append(info['ego_to_world_matrix'])
            meta_dict["Camera_FrontRight"]['cam_id'].append(camera_id_mapping['Camera_FrontRight'])
            meta_dict["Camera_FrontRight"]['intrinsics'].append(
                img_infos[0]["Camera_FrontRight"]["camera_intrinsic_matrix"])
            # check for camera visisblity is
            # meta_dict["Camera_FrontLef"]['kind_of_cam_id'].append(info['camera_visibility'])
            meta_dict["Camera_FrontRight"]['extrinsics'].append(info["lidar_to_ego_matrix"])

            # fourth_camera
            meta_dict["Camera_BackLeft"]['timestamp'].append(img_infos[0]["Camera_BackLeft"]["timestamp"])
            meta_dict["Camera_BackLeft"]["filepath"].append(img_infos[0]["Camera_BackLeft"]["image_path"])
            meta_dict["Camera_BackLeft"]['ego_pose'].append(info['ego_to_world_matrix'])
            meta_dict["Camera_BackLeft"]['cam_id'].append(camera_id_mapping['Camera_BackLeft'])
            meta_dict["Camera_BackLeft"]['intrinsics'].append(
                img_infos[0]["Camera_BackLeft"]["camera_intrinsic_matrix"])
            # check for camera visisblity is
            # meta_dict["Camera_FrontLef"]['kind_of_cam_id'].append(info['camera_visibility'])
            meta_dict["Camera_BackLeft"]['extrinsics'].append(info["lidar_to_ego_matrix"])

            # fifth_camera
            meta_dict["Camera_Back"]['timestamp'].append(img_infos[0]["Camera_Back"]["timestamp"])
            meta_dict["Camera_Back"]["filepath"].append(img_infos[0]["Camera_Back"]["image_path"])
            meta_dict["Camera_Back"]['ego_pose'].append(info['ego_to_world_matrix'])
            meta_dict["Camera_Back"]['cam_id'].append(camera_id_mapping['Camera_Back'])
            meta_dict["Camera_Back"]['intrinsics'].append(img_infos[0]["Camera_Back"]["camera_intrinsic_matrix"])
            # check for camera visisblity is
            # meta_dict["Camera_FrontLef"]['kind_of_cam_id'].append(info['camera_visibility'])
            meta_dict["Camera_Back"]['extrinsics'].append(info["lidar_to_ego_matrix"])
            # sxcth_camera
            meta_dict["Camera_BackRight"]['timestamp'].append(img_infos[0]["Camera_BackRight"]["timestamp"])
            meta_dict["Camera_BackRight"]["filepath"].append(img_infos[0]["Camera_BackRight"]["image_path"])
            meta_dict["Camera_BackRight"]['ego_pose'].append(info['ego_to_world_matrix'])
            meta_dict["Camera_BackRight"]['cam_id'].append(camera_id_mapping['Camera_BackRight'])
            meta_dict["Camera_BackRight"]['intrinsics'].append(
                img_infos[0]["Camera_BackRight"]["camera_intrinsic_matrix"])
            # check for camera visisblity is
            # meta_dict["Camera_FrontLef"]['kind_of_cam_id'].append(info['camera_visibility'])
            meta_dict["Camera_BackRight"]['extrinsics'].append(info["lidar_to_ego_matrix"])

        #print(meta_dict)
        #"""
        # with open("sample.json", "w") as outfile:
        #json.dump(meta_dict, outfile, cls=NumpyEncoder)
        #logger.info(f"[Pixel] Saved camera meta to {self.meta_file_path}")

        #"""

        return meta_dict


    def create_all_filelist(self):
        # NuScenes dataset is not synchronized, so we need to find the minimum shared
        # scene length, and only use the frames within the shared scene length.
        # we also define the start and end timestep within the shared scene length

        num_timestamps = 100000000
        for camera in self.camera_list:
            if len(self.meta_dict[camera]["timestamp"]) < num_timestamps:
                num_timestamps = len(self.meta_dict[camera]["timestamp"])
        logger.info(f"[Pixel] Min shared scene length: {num_timestamps}")
        self.scene_total_num_timestamps = num_timestamps

        if self.end_timestep == -1:
            self.end_timestep = num_timestamps - 1
        else:
            self.end_timestep = min(self.end_timestep, num_timestamps - 1)

        # to make sure the last timestep is included
        self.end_timestep += 1
        self.start_timestep = min(self.start_timestep, self.end_timestep - 1)
        self.scene_fraction = (self.end_timestep - self.start_timestep) / num_timestamps

        logger.info(f"[Pixel] Start timestep: {self.start_timestep}")
        logger.info(f"[Pixel] End timestep: {self.end_timestep}")

        img_filepaths, feat_filepaths, sky_mask_filepaths = [], [], []
        # TODO: support dynamic masks

        for t in range(self.start_timestep, self.end_timestep):
            for cam_idx in self.camera_list:
                img_filepath = os.path.join(
                    self.data_path, self.meta_dict[cam_idx]["filepath"][t]
                )
                img_filepaths.append(img_filepath)
                sky_mask_filepaths.append(
                    img_filepath.replace("samples", "samples_sky_mask")
                    .replace("sweeps", "sweeps_sky_mask")
                    .replace(".jpg", ".png")
                )
                feat_filepaths.append(
                    img_filepath.replace(
                        "samples", f"samples_{self.data_cfg.feature_model_type}"
                    )
                    .replace("sweeps", f"sweeps_{self.data_cfg.feature_model_type}")
                    .replace(".jpg", ".npy")
                )
        self.img_filepaths = np.array(img_filepaths)
        self.sky_mask_filepaths = np.array(sky_mask_filepaths)
        self.feat_filepaths = np.array(feat_filepaths)

    def load_calibrations(self):
        # compute per-image poses and intrinsics
        cam_to_worlds, ego_to_worlds = [], []
        intrinsics, timesteps, cam_ids = [], [], []
        timestamps = []

        # we tranform the camera poses w.r.t. the first timestep to make the origin of
        # the first ego pose  as the origin of the world coordinate system.
        initial_ego_to_global = self.meta_dict["Camera_Front"]["ego_pose"][
            self.start_timestep
        ]
        global_to_initial_ego = np.linalg.inv(initial_ego_to_global)

        for t in range(self.start_timestep, self.end_timestep):

            """
            Ego to world coordinate system for deep accident
            """

            ego_to_global_current = self.meta_dict["Camera_Front"]["ego_pose"][t]
            # compute ego_to_world transformation
            ego_to_world = global_to_initial_ego @ ego_to_global_current
            ego_to_worlds.append(ego_to_world)
            for cam_name in self.camera_list:
                cam_to_ego = self.meta_dict[cam_name]["extrinsics"][t]
                # Because we use opencv coordinate system to generate camera rays,
                # we need to store the transformation from opencv coordinate system to dataset
                # coordinate system. However, the nuScenes dataset uses the same coordinate
                # system as opencv, so we just store the identity matrix.
                # opencv coordinate system: x right, y down, z front
                cam_to_ego = cam_to_ego @ self.OPENCV2DATASET
                cam2world = ego_to_world @ cam_to_ego
                cam_to_worlds.append(cam2world)
                intrinsics.append(self.meta_dict[cam_name]["intrinsics"][t])
                timesteps.append(t)
                cam_ids.append(self.meta_dict[cam_name]["cam_id"][t])
                timestamps.append(
                    self.meta_dict[cam_name]["timestamp"][t]
                    / 1e6
                    * np.ones_like(self.meta_dict[cam_name]["cam_id"][t])
                )

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        # scale the intrinsics according to the load size
        self.intrinsics[..., 0, 0] *= (
            self.data_cfg.load_size[1] / self.ORIGINAL_SIZE[0][1]
        )
        self.intrinsics[..., 1, 1] *= (
            self.data_cfg.load_size[0] / self.ORIGINAL_SIZE[0][0]
        )
        self.intrinsics[..., 0, 2] *= (
            self.data_cfg.load_size[1] / self.ORIGINAL_SIZE[0][1]
        )
        self.intrinsics[..., 1, 2] *= (
            self.data_cfg.load_size[0] / self.ORIGINAL_SIZE[0][0]
        )

        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()
        #self.cam_to_worlds = None
        self.ego_to_worlds = torch.from_numpy(np.stack(ego_to_worlds, axis=0)).float()
        self.global_to_initial_ego = torch.from_numpy(global_to_initial_ego).float()
        self.cam_ids = torch.from_numpy(np.stack(cam_ids, axis=0)).long()

        # the underscore here is important.
        self._timestamps = torch.tensor(timestamps, dtype=torch.float64)
        self._timesteps = torch.from_numpy(np.stack(timesteps, axis=0)).long()
        """
        until here workin but values are different such AS global to worlds scene lenthg and time stap
        """


class DeepaccidentDataset(SceneDataset):
    dataset: str = "nuscenes"

    def __init__(
        self,
        data_cfg: OmegaConf,
    ) -> None:
        super().__init__(data_cfg)
        assert self.data_cfg.dataset == "deepaccident"
        self.data_path = self.data_cfg.data_root

        # ---- create pixel source ---- #
        self.pixel_source, self.lidar_source = self.build_data_source()
        self.aabb = self.get_aabb()

        # ---- define train and test indices ---- #
        (
            self.train_timesteps,
            self.test_timesteps,
            self.train_indices,
            self.test_indices,
        ) = self.split_train_test()
        # ---- create split wrappers ---- #
        pixel_sets, lidar_sets = self.build_split_wrapper()
        self.train_pixel_set, self.test_pixel_set, self.full_pixel_set = pixel_sets
        self.train_lidar_set, self.test_lidar_set, self.full_lidar_set = lidar_sets

    def build_split_wrapper(self):
        """
        Makes each data source as a Pytorch Dataset
        """
        train_pixel_set, test_pixel_set, full_pixel_set = None, None, None
        train_lidar_set, test_lidar_set, full_lidar_set = None, None, None
        assert (
            len(self.test_indices) == 0
        ), "Test split is not supported yet for nuscenes"
        # ---- create split wrappers ---- #
        if self.pixel_source is not None:
            train_pixel_set = SplitWrapper(
                datasource=self.pixel_source,
                # train_indices are img indices, so the length is num_cams * num_train_timesteps
                split_indices=self.train_indices,
                split="train",
                ray_batch_size=self.data_cfg.ray_batch_size,
            )
            full_pixel_set = SplitWrapper(
                datasource=self.pixel_source,
                # cover all the images
                split_indices=np.arange(self.pixel_source.num_imgs).tolist(),
                split="full",
                ray_batch_size=self.data_cfg.ray_batch_size,
            )
        if self.lidar_source is not None:
            train_lidar_set = SplitWrapper(
                datasource=self.lidar_source,
                # the number of image timesteps is different from the number of lidar timesteps
                # TODO: find a better way to handle this
                # currently use all the lidar timesteps for training
                split_indices=np.arange(self.lidar_source.num_timesteps),
                split="train",
                ray_batch_size=self.data_cfg.ray_batch_size,
            )
            full_lidar_set = SplitWrapper(
                datasource=self.lidar_source,
                # cover all the lidar scans
                split_indices=np.arange(self.lidar_source.num_timesteps),
                split="full",
                ray_batch_size=self.data_cfg.ray_batch_size,
            )

        pixel_set = (train_pixel_set, test_pixel_set, full_pixel_set)
        lidar_set = (train_lidar_set, test_lidar_set, full_lidar_set)
        return pixel_set, lidar_set

    def build_data_source(self):
        pixel_source, lidar_source = None, None
        all_timestamps = []
        # ---- create pixel source ---- #
        load_pixel = (
            self.data_cfg.pixel_source.load_rgb
            or self.data_cfg.pixel_source.load_sky_mask
            or self.data_cfg.pixel_source.load_dynamic_mask
            or self.data_cfg.pixel_source.load_feature
        )
        if load_pixel:
            pixel_source = DeepAccidentPixelSource(
                pixel_data_config=self.data_cfg.pixel_source,
                data_path=self.data_path,
                scene_idx=self.scene_idx,
                meta_file_path= self.data_cfg.img_meta_file_path,
                start_timestep=self.data_cfg.start_timestep,
                end_timestep=self.data_cfg.end_timestep,
            )
            pixel_source.to(self.device)
            all_timestamps.append(pixel_source.timestamps)
            self.start_timestep = pixel_source.start_timestep
            self.end_timestep = pixel_source.end_timestep
            self.scene_fraction = pixel_source.scene_fraction
        # ---- create lidar source ---- #
        if self.data_cfg.lidar_source.load_lidar:
            lidar_source = DeepAccidentPixelSource(
                lidar_data_config=self.data_cfg.lidar_source,
                data_path=self.data_path,
                meta_file_path=self.lidar_meta_file_path,
                nusc=pixel_source.nusc if pixel_source is not None else None,
                scene_idx=self.scene_idx,
                start_timestep=self.start_timestep,
                fraction=self.scene_fraction,
                global_to_initial_ego=pixel_source.global_to_initial_ego,
            )
            lidar_source.to(self.device)
            all_timestamps.append(lidar_source.timestamps)

        assert len(all_timestamps) > 0, "No data source is loaded"
        all_timestamps = torch.cat(all_timestamps, dim=0)
        # normalize the timestamps
        all_timestamps = (all_timestamps - all_timestamps.min()) / (
            all_timestamps.max() - all_timestamps.min()
        )
        all_timestamps = all_timestamps.float()
        if pixel_source is not None:
            pixel_source.register_normalized_timestamps(
                all_timestamps[: len(pixel_source.timestamps)]
            )
        if lidar_source is not None:
            lidar_source.register_normalized_timestamps(
                all_timestamps[-len(lidar_source.timestamps) :]
            )
        return pixel_source, lidar_source

    def split_train_test(self):
        assert (
            self.data_cfg.pixel_source.test_image_stride == 0
        ), "test_image_stride > 0 is not supported for nuscenes dataset. "
        if self.data_cfg.pixel_source.test_image_stride != 0:
            test_timesteps = np.arange(
                self.data_cfg.pixel_source.test_image_stride,
                self.num_img_timesteps,
                self.data_cfg.pixel_source.test_image_stride,
            )
        else:
            test_timesteps = []
        train_timesteps = np.array(
            [i for i in range(self.num_img_timesteps) if i not in test_timesteps]
        )
        logger.info(
            f"Train timesteps: \n{np.arange(self.start_timestep, self.end_timestep)[train_timesteps]}"
        )
        logger.info(
            f"Test timesteps: \n{np.arange(self.start_timestep, self.end_timestep)[test_timesteps]}"
        )

        # propagate the train and test timesteps to the train and test indices
        train_indices, test_indices = [], []
        for t in range(self.num_img_timesteps):
            if t in train_timesteps:
                for cam in range(self.pixel_source.num_cams):
                    train_indices.append(t * self.pixel_source.num_cams + cam)
            elif t in test_timesteps:
                for cam in range(self.pixel_source.num_cams):
                    test_indices.append(t * self.pixel_source.num_cams + cam)
        logger.info(f"Number of train indices: {len(train_indices)}")
        logger.info(f"Train indices: {train_indices}")
        logger.info(f"Number of test indices: {len(test_indices)}")
        logger.info(f"Test indices: {test_indices}")

        return train_timesteps, test_timesteps, train_indices, test_indices


    def save_videos(self, video_dict, **kwargs):
        return save_videos(
            render_results=video_dict,
            save_pth=kwargs["save_pth"],
            num_timestamps=kwargs["num_timestamps"],
            keys=kwargs["keys"],
            num_cams=kwargs["num_cams"],
            fps=kwargs["fps"],
            verbose=kwargs["verbose"],
        )


    def render_data_videos(
        self,
        save_pth: str,
        split: str = "full",
        fps: int = 24,
        verbose=True,
    ):
        """
        Render a video of the supervision.
        """
        pixel_dataset, lidar_dataset = None, None
        if split == "full":
            if self.pixel_source is not None:
                pixel_dataset = self.full_pixel_set
            if self.lidar_source is not None:
                lidar_dataset = self.full_lidar_set
        elif split == "train":
            if self.pixel_source is not None:
                pixel_dataset = self.train_pixel_set
            if self.lidar_source is not None:
                lidar_dataset = self.train_lidar_set
        elif split == "test":
            if self.pixel_source is not None:
                pixel_dataset = self.test_pixel_set
            if self.lidar_source is not None:
                lidar_dataset = self.test_lidar_set
        else:
            raise NotImplementedError(f"Split {split} not supported")

        # pixel source
        rgb_imgs, dynamic_objects = [], []
        sky_masks, feature_pca_colors = [], []
        lidar_depths = []

        for i in trange(
            len(pixel_dataset), desc="Rendering supervision videos", dynamic_ncols=True
        ):
            data_dict = pixel_dataset[i]
            if "pixels" in data_dict:
                rgb_imgs.append(data_dict["pixels"].cpu().numpy())
            if "dynamic_masks" in data_dict:
                dynamic_objects.append(
                    (data_dict["dynamic_masks"].unsqueeze(-1) * data_dict["pixels"])
                    .cpu()
                    .numpy()
                )
            if "sky_masks" in data_dict:
                sky_masks.append(data_dict["sky_masks"].cpu().numpy())
            if "features" in data_dict:
                features = data_dict["features"]
                features = features @ self.pixel_source.feat_dimension_reduction_mat
                features = (features - self.pixel_source.feat_color_min) / (
                    self.pixel_source.feat_color_max - self.pixel_source.feat_color_min
                ).clamp(0, 1)
                feature_pca_colors.append(features.cpu().numpy())
            if lidar_dataset is not None:
                # to deal with asynchronized data
                # find the closest lidar scan to the current image in time
                closest_lidar_idx = self.lidar_source.find_closest_timestep(
                    data_dict["normed_timestamps"].flatten()[0]
                )
                data_dict = lidar_dataset[closest_lidar_idx]
                lidar_points = (
                    data_dict["lidar_origins"]
                    + data_dict["lidar_ranges"] * data_dict["lidar_viewdirs"]
                )
                # project lidar points to the image plane
                # TODO: consider making this a function
                intrinsic_4x4 = torch.nn.functional.pad(
                    self.pixel_source.intrinsics[i], (0, 1, 0, 1)
                )
                intrinsic_4x4[3, 3] = 1.0
                lidar2img = intrinsic_4x4 @ self.pixel_source.cam_to_worlds[i].inverse()
                lidar_points = (
                    lidar2img[:3, :3] @ lidar_points.T + lidar2img[:3, 3:4]
                ).T
                depth = lidar_points[:, 2]
                cam_points = lidar_points[:, :2] / (depth.unsqueeze(-1) + 1e-6)
                valid_mask = (
                    (cam_points[:, 0] >= 0)
                    & (cam_points[:, 0] < self.pixel_source.WIDTH)
                    & (cam_points[:, 1] >= 0)
                    & (cam_points[:, 1] < self.pixel_source.HEIGHT)
                    & (depth > 0)
                )
                depth = depth[valid_mask]
                _cam_points = cam_points[valid_mask]
                depth_map = torch.zeros(
                    self.pixel_source.HEIGHT, self.pixel_source.WIDTH
                ).to(self.device)
                depth_map[
                    _cam_points[:, 1].long(), _cam_points[:, 0].long()
                ] = depth.squeeze(-1)
                depth_img = depth_map.cpu().numpy()
                depth_img = depth_visualizer(depth_img, depth_img > 0)
                mask = (depth_map.unsqueeze(-1) > 0).cpu().numpy()
                # show the depth map on top of the rgb image
                image = rgb_imgs[-1] * (1 - mask) + depth_img * mask
                lidar_depths.append(image)

        video_dict = {
            "gt_rgbs": rgb_imgs,
            "stacked": lidar_depths,
            "gt_feature_pca_colors": feature_pca_colors,
            # "gt_sky_masks": sky_masks,
            "gt_dynamic_objects": dynamic_objects,
        }
        video_dict = {k: v for k, v in video_dict.items() if len(v) > 0}
        # use 3 cameras a row if there are 6 cameras
        return self.save_videos(
            video_dict,
            save_pth=save_pth,
            num_timestamps=self.num_img_timesteps,
            keys=video_dict.keys(),
            num_cams=self.pixel_source.num_cams,
            fps=fps,
            verbose=verbose,
        )

################################Rosugh code ################

""" 
       total_camera_list = [
            "Camera_FrontLeft",
            "Camera_Front",
            "Camera_FrontRight",
            "Camera_BackLeft",
            "Camera_Back",
            "Camera_BackRight",
        ]

        meta_dict = {
            camera: {
                "timestamp": [],
                "filepath": [],
                "ego_pose": [],
                "cam_id": [],
                "extrinsics": [],
                "intrinsics": [],
            }
            for i, camera in enumerate(total_camera_list)
        }
        current_camera_data_tokens = {camera: None for camera in total_camera_list}
        first_sample = self.nusc.get("sample", self.scene["first_sample_token"])

        # Get the first info


        # Populate meta_dict with data from each camera
        for i, camera in enumerate(total_camera_list):
            camera_info = info["cams"][camera]

            meta_dict[camera]["cam_id"].append(i)
            meta_dict[camera]["timestamp"].append(camera_info["timestamp"])
            meta_dict[camera]["filepath"].append(camera_info["image_path"])

            # Adjust the following based on your actual data structure
            extrinsic = camera_info["lidar_to_camera_matrix"]
            intrinsic = camera_info["camera_intrinsic_matrix"]

            meta_dict[camera]["extrinsics"].append(extrinsic)
            meta_dict[camera]["intrinsics"].append(intrinsic)

        # Now 'meta_dict' contains the information you need for the first info
        print("================================", meta_dict)
"""
