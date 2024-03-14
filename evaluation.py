import argparse
import json
import logging
import os
import time
from typing import List, Optional

import imageio
import numpy as np
import torch
import torch.utils.data
from omegaconf import OmegaConf
from tqdm import tqdm

import builders
import loss
import utils.misc as misc
import wandb
from datasets import metrics
from datasets.base import SceneDataset
from radiance_fields import DensityField, RadianceField
from radiance_fields.render_utils import render_rays
from radiance_fields.video_utils import render_pixels, save_videos
from third_party.nerfacc_prop_net import PropNetEstimator, get_proposal_requires_grad_fn
from utils.logging import MetricLogger, setup_logging
from utils.visualization_tools import visualize_voxels, visualize_scene_flow

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

# a global list of keys to render,
# comment out the keys you don't want to render or uncomment the keys you want to render
render_keys = [
    "gt_rgbs",
    "rgbs",
    "depths",
    # "median_depths",
    "gt_dino_feats",
    "dino_feats",
    "dynamic_rgbs",
    "dynamic_depths",
    "static_rgbs",
    "static_depths",
    "forward_flows",
    "backward_flows",
    "dynamic_rgb_on_static_dinos",
    "dino_pe",
    "dino_feats_pe_free",
    # "dynamic_dino_on_static_rgbs",
    # "shadow_reduced_static_rgbs",
    # "shadow_only_static_rgbs",
    # "shadows",
    # "gt_sky_masks",
    # "sky_masks",
]


@torch.no_grad()
def do_evaluation(
    step: int = 0,
    cfg: OmegaConf = None,
    model: RadianceField = None,
    proposal_networks: Optional[List[DensityField]] = None,
    proposal_estimator: PropNetEstimator = None,
    dataset: SceneDataset = None,
    args: argparse.Namespace = None,
):
    logger.info("Evaluating on the full set...")
    model.eval()
    proposal_estimator.eval()
    for p in proposal_networks:
        p.eval()

    if cfg.eval.eval_occ:
        assert cfg.data.dataset == "waymo", "only support waymo dataset for now"
        device = model.device
        # use every cfg.eval.occ_annotation_stride frames for training
        train_indices = np.arange(
            0, dataset.num_lidar_timesteps, cfg.eval.occ_annotation_stride
        )
        test_indices = [
            x for x in range(dataset.num_lidar_timesteps) if x not in train_indices
        ]
        # collect centroids and labels
        centroids_bank, label_bank = metrics.collect_centroids(
            train_indices, dataset, model, device
        )
        logger.info("Evaluating Few-shot Occ...")
        occ_metrics = metrics.eval_few_shot_occ(
            test_indices, dataset, model, device, centroids_bank, label_bank
        )
        occ_metrics_file = f"{cfg.log_dir}/metrics/occ_eval_{current_time}.json"
        with open(occ_metrics_file, "w") as f:
            json.dump(occ_metrics, f)
        if args.enable_wandb:
            wandb.log(occ_metrics)
        logger.info(
            f"Few-shot Occupancy evaluation metrics saved to {occ_metrics_file}"
        )
        logger.info("Few-shot Occ Results:")
        logger.info(json.dumps(occ_metrics, indent=4))
        logger.info(
            "===> Note: zero accuracy means no valid points for that class in the scene"
        )
        torch.cuda.empty_cache()

    if cfg.eval.eval_lidar_flow and cfg.nerf.model.head.enable_flow_branch:
        assert cfg.data.dataset == "waymo", "only support waymo dataset for now"
        logger.info("Evaluating Lidar Flow...")
        # use metrics from NSFP
        all_flow_metrics = {
            "EPE3D": [],
            "acc3d_strict": [],
            "acc3d_relax": [],
            "angle_error": [],
            "outlier": [],
        }
        for data_dict in tqdm(
            dataset.full_lidar_set, "Evaluating Lidar Flow", dynamic_ncols=True
        ):
            lidar_flow_class = data_dict["lidar_flow_class"]
            for k, v in data_dict.items():
                # remove invalid flow (the information is from GT)
                data_dict[k] = v[lidar_flow_class != -1]

            if data_dict[k].shape[0] == 0:
                logger.info(f"no valid points, skipping...")
                continue

            if cfg.eval.remove_ground_when_eval_lidar_flow:
                # following the setting in scene flow estimation works
                for k, v in data_dict.items():
                    data_dict[k] = v[~data_dict["lidar_ground"]]
            lidar_points = (
                data_dict["lidar_origins"]
                + data_dict["lidar_ranges"] * data_dict["lidar_viewdirs"]
            )
            normalized_timestamps = data_dict["lidar_normed_timestamps"]
            pred_results = model.query_flow(
                positions=lidar_points,
                normed_timestamps=normalized_timestamps,
            )
            pred_flow = pred_results["forward_flow"]
            # flow is only valid when the point is not static
            pred_flow[pred_results["dynamic_density"] < 0.2] *= 0
            # metrics in NSFP
            flow_metrics = metrics.compute_scene_flow_metrics(
                pred_flow[None, ...], data_dict["lidar_flow"][None, ...]
            )
            for k, v in flow_metrics.items():
                all_flow_metrics[k].append(v)
        logger.info("Lidar Flow Results:")
        avg_flow_metrics = {k: np.mean(v) for k, v in all_flow_metrics.items()}
        logger.info(json.dumps(avg_flow_metrics, indent=4))
        flow_metrics_file = f"{cfg.log_dir}/metrics/flow_eval_{current_time}.json"
        with open(flow_metrics_file, "w") as f:
            json.dump(avg_flow_metrics, f)
        logger.info(f"Flow estimation evaluation metrics saved to {flow_metrics_file}")
        if args.enable_wandb:
            wandb.log(avg_flow_metrics)
        torch.cuda.empty_cache()

    if cfg.data.pixel_source.load_rgb and cfg.render.render_low_res:
        logger.info("Rendering full set but in a low_resolution...")
        dataset.pixel_source.update_downscale_factor(1 / cfg.render.low_res_downscale)
        render_results = render_pixels(
            cfg=cfg,
            model=model,
            proposal_networks=proposal_networks,
            proposal_estimator=proposal_estimator,
            dataset=dataset.full_pixel_set,
            compute_metrics=True,
            return_decomposition=True,
        )
        dataset.pixel_source.reset_downscale_factor()
        if args.render_video_postfix is None:
            video_output_pth = os.path.join(cfg.log_dir, "lowres_videos", f"{step}.mp4")
        else:
            video_output_pth = os.path.join(
                cfg.log_dir,
                "lowres_videos",
                f"{step}_{args.render_video_postfix}.mp4",
            )
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            num_timestamps=dataset.num_img_timesteps,
            keys=render_keys,
            save_seperate_video=cfg.logging.save_seperate_video,
            num_cams=dataset.pixel_source.num_cams,
            fps=cfg.render.fps,
            verbose=True,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({f"pixel_rendering/lowres_full/{k}": wandb.Image(v)})

        del render_results, vis_frame_dict
        torch.cuda.empty_cache()

    if cfg.data.pixel_source.load_rgb:
        logger.info("Evaluating Pixels...")
        if dataset.test_pixel_set is not None and cfg.render.render_test:
            logger.info("Evaluating Test Set Pixels...")
            render_results = render_pixels(
                cfg=cfg,
                model=model,
                proposal_estimator=proposal_estimator,
                dataset=dataset.test_pixel_set,
                proposal_networks=proposal_networks,
                compute_metrics=True,
                return_decomposition=True,
            )
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "feat_psnr",
                    "masked_psnr",
                    "masked_ssim",
                    "masked_feat_psnr",
                ]:
                    eval_dict[f"pixel_metrics/test/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            test_metrics_file = f"{cfg.log_dir}/metrics/images_test_{current_time}.json"
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

            if args.render_video_postfix is None:
                video_output_pth = f"{cfg.log_dir}/test_videos/{step}.mp4"
            else:
                video_output_pth = (
                    f"{cfg.log_dir}/test_videos/{step}_{args.render_video_postfix}.mp4"
                )
            vis_frame_dict = save_videos(
                render_results,
                video_output_pth,
                num_timestamps=dataset.num_test_timesteps,
                keys=render_keys,
                num_cams=dataset.pixel_source.num_cams,
                save_seperate_video=cfg.logging.save_seperate_video,
                fps=cfg.render.fps,
                verbose=True,
                # save_images=True,
            )
            if args.enable_wandb:
                for k, v in vis_frame_dict.items():
                    wandb.log({"pixel_rendering/test/" + k: wandb.Image(v)})
            del render_results, vis_frame_dict
            torch.cuda.empty_cache()
        if cfg.render.render_full:
            logger.info("Evaluating Full Set...")
            render_results = render_pixels(
                cfg=cfg,
                model=model,
                proposal_estimator=proposal_estimator,
                dataset=dataset.full_pixel_set,
                proposal_networks=proposal_networks,
                compute_metrics=True,
                return_decomposition=True,
            )
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "feat_psnr",
                    "masked_psnr",
                    "masked_ssim",
                    "masked_feat_psnr",
                ]:
                    eval_dict[f"pixel_metrics/full/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            test_metrics_file = f"{cfg.log_dir}/metrics/images_full_{current_time}.json"
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

            if args.render_video_postfix is None:
                video_output_pth = f"{cfg.log_dir}/full_videos/{step}.mp4"
            else:
                video_output_pth = (
                    f"{cfg.log_dir}/full_videos/{step}_{args.render_video_postfix}.mp4"
                )
            vis_frame_dict = save_videos(
                render_results,
                video_output_pth,
                num_timestamps=dataset.num_img_timesteps,
                keys=render_keys,
                num_cams=dataset.pixel_source.num_cams,
                save_seperate_video=cfg.logging.save_seperate_video,
                fps=cfg.render.fps,
                verbose=True,
            )
            if args.enable_wandb:
                for k, v in vis_frame_dict.items():
                    wandb.log({"pixel_rendering/full/" + k: wandb.Image(v)})
            del render_results, vis_frame_dict
            torch.cuda.empty_cache()
        # TODO: add a novel trajectory rendering part






