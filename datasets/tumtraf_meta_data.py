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


root_path = "/home/uchihadj/TUMtraf/robust-dynrf/dataset/custom/sequence_01/images"
meta_data_ego_vehicle = ("/home/uchihadj/TUMtraf/tum-traffic-dataset-dev-kit/calib/s110_camera_basler_south2_8mm.json")
town_name = "1688625741_027764001_s110_camera_basler_south2_8mm"


timestamp=0
# Initialize the meta_dict
meta_dict = {

    "Camera_Front": {"timestamp": [], "filepath": [], "ego_pose": [], "cam_id": [], "extrinsics": [],
                     "intrinsics": []},

}
# meta_dict["CAMERA_FRONTLEFT"]['timestamp'].append(img_infos["Camera_FrontLe"])
# Map camera names to IDs
camera_id_mapping = {
    "Camera_Front": 0,

}

# Iterate through the files in the folder
for filename in os.listdir(root_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more file extensions if necessary
        file_path = os.path.join(root_path, filename)
        # Append the file path to the metadata dictionary
        meta_dict["Camera_Front"]["filepath"].append(file_path)
        meta_dict["Camera_Front"]["cam_id"].append(camera_id_mapping["Camera_Front"])
        meta_dict["Camera_Front"]["timestamp"].append(timestamp)
        timestamp +=1
        with open(meta_data_ego_vehicle, 'rb') as file:
            calib_data = json.load(file)

        intrinsic_Camera_Front = calib_data.get('calibrated_intrinsic_camera_matrix', None)
        meta_dict["Camera_Front"]["intrinsics"].append(intrinsic_Camera_Front)
        rotation_matrix = calib_data.get('rotation_matrix', None)
        translation_matrix = calib_data.get("translation_matrix", None)
        extrinsic_matrix = np.zeros((4, 4))
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = translation_matrix
        extrinsic_matrix[3, 3] = 1
        meta_dict["Camera_Front"]["extrinsics"].append(extrinsic_matrix)
        meta_dict["Camera_Front"]["ego_pose"].append(extrinsic_matrix)



with open("tumtraf_meta_data_scene_2.json", "w") as outfile:
   json.dump(meta_dict, outfile, cls=NumpyEncoder)

print("Finished")





