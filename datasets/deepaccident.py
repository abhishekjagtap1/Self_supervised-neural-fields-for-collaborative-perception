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

from datasets.base.lidar_source import SceneLidarSource
from datasets.base.pixel_source import ScenePixelSource
from datasets.base.scene_dataset import SceneDataset
from datasets.base.split_wrapper import SplitWrapper
from datasets.utils import voxel_coords_to_world_coords
from radiance_fields.video_utils import save_videos, depth_visualizer
from utils.misc import NumpyEncoder
class Deepaccident(SceneDataset):
    dataset: str = "nuscenes"

    def __init__(
        self,
        data_cfg: OmegaConf,
    ) -> None:
        super().__init__(data_cfg)
        assert self.data_cfg.dataset == "nuscenes"
        self.data_path = self.data_cfg.data_root
        self.processed_data_path = os.path.join(
            self.data_path, "emernerf_metas", f"{self.scene_idx:03d}"
        )
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
        self.img_meta_file_path = os.path.join(
            self.processed_data_path, "img_meta.json"
        )
        self.lidar_meta_file_path = os.path.join(
            self.processed_data_path, "lidar_meta.json"
        )

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
            pixel_source = NuScenesPixelSource(
                pixel_data_config=self.data_cfg.pixel_source,
                data_path=self.data_path,
                scene_idx=self.scene_idx,
                meta_file_path=self.img_meta_file_path,
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
            lidar_source = NuScenesLiDARSource(
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