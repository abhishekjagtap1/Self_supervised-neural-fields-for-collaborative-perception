import json
import numpy as np
import pickle
import argparse
import os
#from utils.misc import NumpyEncoder

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # convert numpy array to list
        return super(NumpyEncoder, self).default(obj)


root_path = "/home/uchihadj/DeepAccident/data/DeepAccident_data/type1_subtype1_accident/ego_vehicle"
meta_data_ego_vehicle = ("/home/uchihadj/DeepAccident/data/DeepAccident_data/"
                         "type1_subtype1_accident/ego_vehicle/calib/Town01_type001_subtype0001_scenario00001")
town_name = "Town01_type001_subtype0001_scenario00001"



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
# meta_dict["CAMERA_FRONTLEFT"]['timestamp'].append(img_infos["Camera_FrontLe"])
# Map camera names to IDs
camera_id_mapping = {
    "Camera_FrontLeft": 0,
    "Camera_Front": 1,
    "Camera_FrontRight": 2,
    "Camera_BackLeft": 3,
    "Camera_Back": 4,
    "Camera_BackRight": 5
}

for camera_name in meta_dict:
    camera_path = os.path.join(root_path, camera_name, town_name)
    calib_path = os.path.join(root_path, "calib", town_name)
    for i in range(1, 97):
        image_name = f"{town_name}_{str(i).zfill(3)}.jpg"
        image_path = os.path.join(camera_path, image_name)
        meta_dict[camera_name]["filepath"].append(image_path)
        meta_dict[camera_name]["cam_id"].append(camera_id_mapping[camera_name])
        meta_dict[camera_name]["timestamp"].append(i)


#print(meta_dict)
calib_path = os.path.join(root_path, "calib", town_name)
for i in range(1, 97):
    ##Camera intrinscis and extrinsics
    calib_file_path = os.path.join(calib_path, f"{town_name}_{str(i).zfill(3)}.pkl")
    #print(calib_file_path)
    with open(calib_file_path, 'rb') as file:
        calib_data = pickle.load(file)
    """
    Get camera intrrinsic matrix for each camerra
    """
    intrinsic_Camera_FrontLeft = calib_data.get('intrinsic_Camera_FrontLeft', None)
    intrinsic_Camera_Front = calib_data.get('intrinsic_Camera_Front', None)
    intrinsic_Camera_FrontRight = calib_data.get('intrinsic_Camera_FrontRight',None)
    intrinsic_Camera_BackLeft = calib_data.get("intrinsic_Camera_BackLeft", None)
    intrinsic_Camera_Back = calib_data.get("intrinsic_Camera_Back", None)
    intrinsic_Camera_BackRight = calib_data.get("intrinsic_Camera_BackRight", None)

    """
    Similarly get Extrinsics for each Camera
    """
    lidar_to_Camera_FrontLeft = calib_data.get("lidar_to_Camera_FrontLeft", None)
    lidar_to_Camera_Front= calib_data.get("lidar_to_Camera_Front", None)
    lidar_to_Camera_FrontRight = calib_data.get("lidar_to_Camera_FrontRight", None)
    lidar_to_Camera_BackLeft = calib_data.get("lidar_to_Camera_BackLeft", None)
    lidar_to_Camera_Back = calib_data.get("lidar_to_Camera_Back", None)
    lidar_to_Camera_BackRight = calib_data.get("lidar_to_Camera_BackRight", None)

    """
    Dumb way to append to meta_dict
    """

    meta_dict["Camera_FrontLeft"]["intrinsics"].append(intrinsic_Camera_FrontLeft)
    meta_dict["Camera_Front"]["intrinsics"].append(intrinsic_Camera_Front)
    meta_dict["Camera_FrontRight"]["intrinsics"].append(intrinsic_Camera_FrontRight)
    meta_dict["Camera_BackLeft"]["intrinsics"].append(intrinsic_Camera_BackLeft)
    meta_dict["Camera_Back"]["intrinsics"].append(intrinsic_Camera_Back)
    meta_dict["Camera_BackRight"]["intrinsics"].append(intrinsic_Camera_BackRight)

    meta_dict["Camera_FrontLeft"]["extrinsics"].append(lidar_to_Camera_FrontLeft)
    meta_dict["Camera_Front"]["extrinsics"].append(lidar_to_Camera_Front)
    meta_dict["Camera_FrontRight"]["extrinsics"].append(lidar_to_Camera_FrontRight)
    meta_dict["Camera_BackLeft"]["extrinsics"].append(lidar_to_Camera_BackLeft)
    meta_dict["Camera_Back"]["extrinsics"].append(lidar_to_Camera_Back)
    meta_dict["Camera_BackRight"]["extrinsics"].append(lidar_to_Camera_BackRight)

    """
    Dumb waay of appending ego_pose
    """
    ego_pose_meta = calib_data.get("ego_to_world", None)
    meta_dict["Camera_FrontLeft"]["ego_pose"].append(ego_pose_meta)
    meta_dict["Camera_Front"]["ego_pose"].append(ego_pose_meta)
    meta_dict["Camera_FrontRight"]["ego_pose"].append(ego_pose_meta)
    meta_dict["Camera_BackLeft"]["ego_pose"].append(ego_pose_meta)
    meta_dict["Camera_Back"]["ego_pose"].append(ego_pose_meta)
    meta_dict["Camera_BackRight"]["ego_pose"].append(ego_pose_meta)

with open("v2x_ego_vehicle.json", "w") as outfile:
   json.dump(meta_dict, outfile, cls=NumpyEncoder)

print("Finished")
