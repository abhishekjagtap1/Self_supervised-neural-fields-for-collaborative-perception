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



root_path_1 = "/home/uchihadj/TUMtraf/tumtraf_v2x_cooperative_perception_dataset/val/images/vehicle_camera_basler_16mm"
calib_path_1 = "/home/uchihadj/TUMtraf/tum-traffic-dataset-dev-kit/calib/vehicle_camera_basler_16mm.json"


root_path_2 = "/home/uchihadj/TUMtraf/tumtraf_v2x_cooperative_perception_dataset/val/images/s110_camera_basler_east_8mm"
calib_path_2 = "/home/uchihadj/TUMtraf/tum-traffic-dataset-dev-kit/calib/s110_camera_basler_east_8mm.json"

root_path_3 = "/home/uchihadj/TUMtraf/tumtraf_v2x_cooperative_perception_dataset/val/images/s110_camera_basler_north_8mm"
calib_path_3 = "/home/uchihadj/TUMtraf/tum-traffic-dataset-dev-kit/calib/s110_camera_basler_north_8mm.json"

root_path_4 = "/home/uchihadj/TUMtraf/tumtraf_v2x_cooperative_perception_dataset/val/images/s110_camera_basler_south1_8mm"
calib_path_4 = "/home/uchihadj/TUMtraf/tum-traffic-dataset-dev-kit/calib/s110_camera_basler_south1_8mm.json"

root_path_5 = "/home/uchihadj/TUMtraf/tumtraf_v2x_cooperative_perception_dataset/val/images/s110_camera_basler_south2_8mm"
calib_path_5 = "/home/uchihadj/TUMtraf/tum-traffic-dataset-dev-kit/calib/s110_camera_basler_south2_8mm.json"

#town_name = "1688625741_027764001_s110_camera_basler_south2_8mm"


timestamp=0
# Initialize the meta_dict
meta_dict = {
    "Camera_FrontLeft": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
                         "intrinsics": []},
    "Camera_Front": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
                     "intrinsics": []},
    "Camera_FrontRight": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
                          "intrinsics": []},
    "Camera_Back": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
                    "intrinsics": []},
    "Camera_BackLeft": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
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

}

# Iterate through the files in the folder
for filename in os.listdir(root_path_1):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more file extensions if necessary
        file_path = os.path.join(root_path_1, filename)
        # Append the file path to the metadata dictionary
        meta_dict["Camera_Front"]["filepath"].append(file_path)
        meta_dict["Camera_Front"]["cam_id"].append(camera_id_mapping["Camera_Front"])
        meta_dict["Camera_Front"]["timestamp"].append(timestamp)
        timestamp +=1
        with open(calib_path_1, 'rb') as file:
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
        del intrinsic_Camera_Front

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
        meta_dict["Camera_FrontRight"]["filepath"].append(file_path)
        meta_dict["Camera_FrontRight"]["cam_id"].append(camera_id_mapping["Camera_FrontRight"])
        meta_dict["Camera_FrontRight"]["timestamp"].append(timestamp)
        timestamp +=1
        with open(calib_path_5, 'rb') as file:
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
timestamp=0
# Iterate through the files in the folder
for filename in os.listdir(root_path_2):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more file extensions if necessary
        file_path = os.path.join(root_path_2, filename)
        # Append the file path to the metadata dictionary
        meta_dict["Camera_BackLeft"]["filepath"].append(file_path)
        meta_dict["Camera_BackLeft"]["cam_id"].append(camera_id_mapping["Camera_BackLeft"])
        meta_dict["Camera_BackLeft"]["timestamp"].append(timestamp)
        timestamp +=1
        with open(calib_path_2, 'rb') as file:
            calib_data = json.load(file)

        #intrinsic_Camera_Front = calib_data.get('calibrated_intrinsic_camera_matrix', None)
        intrinsic_Camera_Front = calib_data.get('optimal_intrinsic_camera_matrix', None)
        meta_dict["Camera_BackLeft"]["intrinsics"].append(intrinsic_Camera_Front)
        rotation_matrix = calib_data.get('rotation_matrix', None)
        translation_matrix = calib_data.get("translation_matrix", None)
        extrinsic_matrix = np.zeros((4, 4))
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation_matrix
        extrinsic_matrix[3, 3] = 1
        meta_dict["Camera_BackLeft"]["extrinsics"].append(extrinsic_matrix)
        meta_dict["Camera_BackLeft"]["ego_pose"].append(extrinsic_matrix)
timestamp=0
# Iterate through the files in the folder
for filename in os.listdir(root_path_3):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more file extensions if necessary
        file_path = os.path.join(root_path_3, filename)
        # Append the file path to the metadata dictionary
        meta_dict["Camera_Back"]["filepath"].append(file_path)
        meta_dict["Camera_Back"]["cam_id"].append(camera_id_mapping["Camera_Back"])
        meta_dict["Camera_Back"]["timestamp"].append(timestamp)
        timestamp +=1
        with open(calib_path_3, 'rb') as file:
            calib_data = json.load(file)

        #intrinsic_Camera_Front = calib_data.get('calibrated_intrinsic_camera_matrix', None)
        intrinsic_Camera_Front = calib_data.get('optimal_intrinsic_camera_matrix', None)
        meta_dict["Camera_Back"]["intrinsics"].append(intrinsic_Camera_Front)
        rotation_matrix = calib_data.get('rotation_matrix', None)
        translation_matrix = calib_data.get("translation_matrix", None)
        extrinsic_matrix = np.zeros((4, 4))
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation_matrix
        extrinsic_matrix[3, 3] = 1
        meta_dict["Camera_Back"]["extrinsics"].append(extrinsic_matrix)
        meta_dict["Camera_Back"]["ego_pose"].append(extrinsic_matrix)


with open("tumtraf_collaborative_val_data.json", "w") as outfile:
   json.dump(meta_dict, outfile, cls=NumpyEncoder)

print("Finished")





