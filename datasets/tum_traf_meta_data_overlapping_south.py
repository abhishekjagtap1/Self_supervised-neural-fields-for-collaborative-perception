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

root_path_4 = "/home/uchihadj/TUMtraf/robust-dynrf/dataset/custom/Overlapping_scene_South_1and_2/south_2"
calib_path_4 = "/home/uchihadj/TUMtraf/tum-traffic-dataset-dev-kit/calib/s110_camera_basler_south2_8mm.json"


root_path_5 = "/home/uchihadj/TUMtraf/robust-dynrf/dataset/custom/Overlapping_scene_South_1and_2/south_1"
calib_path_5 = "/home/uchihadj/TUMtraf/tum-traffic-dataset-dev-kit/calib/s110_camera_basler_south1_8mm.json"
"front_madu_above"

root_path_3 = "/home/uchihadj/TUMtraf/robust-dynrf/dataset/custom/Overlapping_scene_South_1and_2/south_3"
calib_path_3 = "/home/uchihadj/TUMtraf/tum-traffic-dataset-dev-kit/calib/s110_camera_basler_north_8mm.json"


timestamp=0
# Initialize the meta_dict
meta_dict = {
    "Camera_FrontLeft": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
                         "intrinsics": []},
    "Camera_Front": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
                     "intrinsics": []},
    "Camera_FrontRight": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
                          "intrinsics": []}

}
# meta_dict["CAMERA_FRONTLEFT"]['timestamp'].append(img_infos["Camera_FrontLe"])
# Map camera names to IDs
camera_id_mapping = {
    "Camera_FrontLeft": 0,
    "Camera_Front": 1,
    "Camera_FrontRight": 2

}


timestamp=0
# Iterate through the files in the folder
for filename in os.listdir(root_path_4):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more file extensions if necessary
        file_path = os.path.join(root_path_4, filename)
        # Append the file path to the metadata dictionary
        meta_dict["Camera_FrontLeft"]["filepath"].append(file_path)
        meta_dict["Camera_FrontLeft"]["cam_id"].append(camera_id_mapping["Camera_FrontLeft"])
        meta_dict["Camera_FrontLeft"]["timestamp"].append(timestamp)
        timestamp +=1
        with open(calib_path_4, 'rb') as file:
            calib_data = json.load(file)

        #intrinsic_Camera_Front = calib_data.get('calibrated_intrinsic_camera_matrix', None)
        intrinsic_Camera_Front = calib_data.get('optimal_intrinsic_camera_matrix', None)
        meta_dict["Camera_FrontLeft"]["intrinsics"].append(intrinsic_Camera_Front)
        rotation_matrix = calib_data.get('rotation_matrix', None)
        translation_matrix = calib_data.get("translation_matrix", None)
        extrinsic_matrix = np.zeros((4, 4))
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation_matrix
        extrinsic_matrix[3, 3] = 1
        meta_dict["Camera_FrontLeft"]["extrinsics"].append(extrinsic_matrix)
        meta_dict["Camera_FrontLeft"]["ego_pose"].append(extrinsic_matrix)

timestamp=0
# Iterate through the files in the folder
for filename in os.listdir(root_path_5):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more file extensions if necessary
        file_path = os.path.join(root_path_5, filename)
        # Append the file path to the metadata dictionary
        meta_dict["Camera_Front"]["filepath"].append(file_path)
        meta_dict["Camera_Front"]["cam_id"].append(camera_id_mapping["Camera_Front"])
        meta_dict["Camera_Front"]["timestamp"].append(timestamp)
        timestamp +=1
        with open(calib_path_5, 'rb') as file:
            calib_data = json.load(file)

        #intrinsic_Camera_Front = calib_data.get('calibrated_intrinsic_camera_matrix', None)
        intrinsic_Camera_Front = calib_data.get('optimal_intrinsic_camera_matrix', None)
        meta_dict["Camera_Front"]["intrinsics"].append(intrinsic_Camera_Front)
        rotation_matrix = calib_data.get('rotation_matrix', None)
        translation_matrix = calib_data.get("translation_matrix", None)
        extrinsic_matrix = np.zeros((4, 4))
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation_matrix
        extrinsic_matrix[3, 3] = 1
        meta_dict["Camera_Front"]["extrinsics"].append(extrinsic_matrix)
        meta_dict["Camera_Front"]["ego_pose"].append(extrinsic_matrix)

timestamp=0
# Iterate through the files in the folder
for filename in os.listdir(root_path_3):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more file extensions if necessary
        file_path = os.path.join(root_path_3, filename)
        # Append the file path to the metadata dictionary
        meta_dict["Camera_FrontRight"]["filepath"].append(file_path)
        meta_dict["Camera_FrontRight"]["cam_id"].append(camera_id_mapping["Camera_FrontRight"])
        meta_dict["Camera_FrontRight"]["timestamp"].append(timestamp)
        timestamp +=10
        with open(calib_path_3, 'rb') as file:
            calib_data = json.load(file)

        #intrinsic_Camera_Front = calib_data.get('calibrated_intrinsic_camera_matrix', None)
        intrinsic_Camera_Front = calib_data.get('optimal_intrinsic_camera_matrix', None)
        meta_dict["Camera_FrontRight"]["intrinsics"].append(intrinsic_Camera_Front)
        rotation_matrix = calib_data.get('rotation_matrix', None)
        translation_matrix = calib_data.get("translation_matrix", None)
        extrinsic_matrix = np.zeros((4, 4))
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation_matrix
        extrinsic_matrix[3, 3] = 1
        meta_dict["Camera_FrontRight"]["extrinsics"].append(extrinsic_matrix)
        meta_dict["Camera_FrontRight"]["ego_pose"].append(extrinsic_matrix)

with open("tumtraf_overlapping_scene_south_1_2_3.json", "w") as outfile:
   json.dump(meta_dict, outfile, cls=NumpyEncoder)

print("Finished")