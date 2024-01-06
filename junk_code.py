################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
""""
or _, cam_name in enumerate(camera_id_mapping):
    for cam_info in info['cams'].items():
        meta_dict[cam_name]['timestamp'].append(cam_info[1]['timestamp'])
        meta_dict[cam_name]['filepath'].append(cam_info[1]['image_path'])
        meta_dict[cam_name]['ego_pose'].append(info['ego_to_world_matrix'])
        meta_dict[cam_name]['extrinsics'].append(cam_info[1]['lidar_to_camera_matrix'])
        meta_dict[cam_name]['intrinsics'].append(cam_info[1]['camera_intrinsic_matrix'])
        # meta_dict[cam_name]['cam_id'].append(camera_id_mapping['cam_name'])

# Display the result
print(meta_dict)
# Loop through the data and populate the meta_dict
for camera in camera_id_mapping:
    cam_id = camera_id_mapping[camera]
    print(cam_id)
    timestamp = camera_info["timestamp"]

for camera_name, camera_info in info.items():
    # cam_id = camera_id_mapping[camera_name]

    # Extract relevant information
    timestamp = camera_info["timestamp"]
    filepath = camera_info["image_path"]
    # ego_pose = info["ego_pose"]
    extrinsics = camera_info["lidar_to_camera_matrix"]
    intrinsics = camera_info["camera_intrinsic_matrix"]

    # Append to meta_dict
    meta_dict[camera_name]["timestamp"].append(timestamp)
    meta_dict[camera_name]["filepath"].append(filepath)
    # meta_dict[camera_name]["ego_pose"].append(ego_pose)
    # meta_dict[camera_name]["cam_id"].append(cam_id)
    meta_dict[camera_name]["extrinsics"].append(extrinsics)
    meta_dict[camera_name]["intrinsics"].append(intrinsics)

# Convert lists to numpy arrays
for camera_name in meta_dict:
    for key in meta_dict[camera_name]:
        meta_dict[camera_name][key] = np.array(meta_dict[camera_name][key])

# Now meta_dict is in the desired format


# Save the meta_dict to a file
with open("your_meta_file.json", "w") as f:
    json.dump(meta_dict, f, cls=NumpyEncoder)

input_dict = dict(
    scenario_length=info['scenario_length'],
    sample_idx=info['lidar_prefix'],
    pts_filename=info['lidar_path'],
    timestamp=info['timestamp'],
    data_root=self.data_path,
    bev_path=info['bev_path'],
    lidar_to_ego_matrix=info['lidar_to_ego_matrix'],
    ego_to_world_matrix=info['ego_to_world_matrix'],
)
img_infos = []
img_infos.append(info['cams'])
img_infos = input_dict['img_info']

lidar_to_ego_matrix = info['lidar_to_ego_matrix'].astype(np.float32)
lidar2ego_translation = lidar_to_ego_matrix[:3, 3]
lidar2ego_rotation = lidar_to_ego_matrix[:3, :3]

input_dict['lidar2ego_rots'] = torch.tensor(lidar2ego_rotation)
input_dict['lidar2ego_trans'] = torch.tensor(lidar2ego_translation)

# print(input_dict)
######################################################################################
total_camera_list = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
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

# ---- get the first sample of each camera ---- #

current_camera_data_tokens = {camera: None for camera in total_camera_list}
# first_sample = self.nusc.get("sample", self.scene["first_sample_token"])
# for frame_id, img_info in enumerate(img_infos):


for camera in total_camera_list:
    # meta_dict[camera]["cam_id"].append(i)
    meta_dict[camera]["timestamp"].append(info["timestamp"])
    meta_dict[camera]["filepath"].append(info["image_path"])

    current_camera_data_tokens[camera] = first_sample["data"][camera]

while not all(token == "" for token in current_camera_data_tokens.values()):
    for i, camera in enumerate(total_camera_list):
        # skip if the current camera data token is empty
        if current_camera_data_tokens[camera] == "":
            continue

        current_camera_data = self.nusc.get(
            "sample_data", current_camera_data_tokens[camera]
        )

        # ---- timestamp and cam_id ---- #
        meta_dict[camera]["cam_id"].append(i)
        meta_dict[camera]["timestamp"].append(current_camera_data["timestamp"])
        meta_dict[camera]["filepath"].append(current_camera_data["filename"])

        # ---- intrinsics and extrinsics ---- #
        calibrated_sensor_record = self.nusc.get(
            "calibrated_sensor", current_camera_data["calibrated_sensor_token"]
        )
        # intrinsics
        intrinsic = calibrated_sensor_record["camera_intrinsic"]
        meta_dict[camera]["intrinsics"].append(np.array(intrinsic))

        # extrinsics
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = Quaternion(
            calibrated_sensor_record["rotation"]
        ).rotation_matrix
        extrinsic[:3, 3] = np.array(calibrated_sensor_record["translation"])
        meta_dict[camera]["extrinsics"].append(extrinsic)

        # ---- ego pose ---- #
        ego_pose_record = self.nusc.get(
            "ego_pose", current_camera_data["ego_pose_token"]
        )
        ego_pose = np.eye(4)
        ego_pose[:3, :3] = Quaternion(
            ego_pose_record["rotation"]
        ).rotation_matrix
        ego_pose[:3, 3] = np.array(ego_pose_record["translation"])
        meta_dict[camera]["ego_pose"].append(ego_pose)

        current_camera_data_tokens[camera] = current_camera_data["next"]

return input_dict
"""



