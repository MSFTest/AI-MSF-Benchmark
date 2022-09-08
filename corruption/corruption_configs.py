''''
@Project: AI-MSF-benchmark
@Description: Please add Description       
@Time:2022/9/6 14:05       
@Author:NianGao    
 
'''

from enum import Enum, unique


class Modality(Enum):
    C = "C"
    L = "L"
    CL = "CL"
    A_SM = "SM"
    A_TM = "TM"


corrupions_abbr_map = {
    # weather cottuption
    "RN": "rain",  # Rain
    "FG": "fog",  # Fog
    "BR": "brightness",  # Brightness
    "DK": "darkness",  # Darknes

    # sensor cottuption
    "DT": "distortion",  # Distortion
    "MB": "motion_blur",  # Motion blur
    "DB": "defocus_blur",  # Defocus Blur

    # noise cottuption
    "GN_C": "gaussian_noise",  # Image Gaussion Nois
    "GN_L": "gaussian_noise",  # Point Cloud Gaussion Noise
    "IN_C": "impulse_noise",  # Image Impulse Noise
    "IN_L": "impulse_noise",  # Point Cloud Impulse Noise

    # sensor misalignment
    "SM_X": "rotation_x",  # Spatial misalignment i.e. calib error
    "SM_Y": "rotation_y",  # Spatial misalignment i.e. calib error
    "SM_Z": "rotation_z",  # Spatial misalignment i.e. calib error
    "SD": "delay",  # Temporal misalignment i.e. signal delay

    # signal loss
    "LP_C": "loss_partial",  # signal loss partial
    "LP_L": "loss_partial",  # signal loss partial
    "LC_C": "loss_complete",  # signal loss complete
    "LC_L": "loss_complete",  # signal loss complete
}

modality_map = {
    Modality.C: ["BR", "DK", "DT", "MB", "DB", "GN_C", "IN_C", "LP_C", "LC_C"],
    Modality.L: ["GN_L", "IN_L", "LP_L", "LC_L"],
    Modality.CL: ["RN", "FG"],
    Modality.A_SM: ["SM_X", "SM_Y", "SM_Z"],
    Modality.A_TM: ["SD"]
}

# weather_corrupions_map = {
#     "BR": corrupions_map["BR"],
#     "DK": corrupions_map["DK"],
#     "RN": ...,  # Rain
#     "FG": ...,  # Fog
# }
#
# sensor_cottuption_map = {
#     "DT": corrupions_map["DT"],
#     "MB": corrupions_map["MB"],
#     "DB": corrupions_map["DB"],
# }
#
# noise_corrupions_map = {
#     "GN_C": corrupions_map["GN_C"],
#     "GN_L": corrupions_map["GN_L"],
#     "IN_C": corrupions_map["IN_C"],
#     "IN_L": corrupions_map["IN_L"],
# }
#
# sensor_misalignment_map = {
#     "SM_X": corrupions_map["SM_X"],
#     "SM_Y": corrupions_map["SM_Y"],
#     "SM_Z": corrupions_map["SM_Z"],
#     "SD": corrupions_map["SD"],
# }

# weather_corrupions_map = {
#     "BR": image_ops["brightness"],
#     "DK": image_ops["darkness"],
#     "RN": ...,  # Rain
#     "FG": ...,  # Fog
# }
#
# sensor_cottuption_map = {
#     "DT": image_ops["distortion"],
#     "MB": image_ops["motion_blur"],
#     "DB": image_ops["defocus_blur"],
# }
#
# noise_corrupions_map = {
#     "GN_C": image_ops["gaussian_noise"],
#     "GN_L": lidar_ops["gaussian_noise"],
#     "IN_C": image_ops["impulse_noise"],
#     "IN_L": lidar_ops["impulse_noise"],
# }
#
# sensor_misalignment_map = {
#     "SM_X": image_ops["rotation_x"],
#     "SM_Y": lidar_ops["rotation_y"],
#     "SM_Z": image_ops["rotation_z"],
#     "SD": lidar_ops["delay"],
# }
