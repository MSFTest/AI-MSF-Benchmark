''''
@Project: fusion_rewrite   
@Description: Please add Description       
@Time:2022/9/6 14:24       
@Author:NianGao    
 
'''
''''
@Project: app
@Description: Please add Description
@Time:2022/1/13 17:26
@Author:NianGao

'''
import functools
import math
import numpy as np


class CalibOperator(object):
    """
    :Author:  NianGao
    :Create:  2022/1/13
    """

    @staticmethod
    def get_dop_method_arr():
        dop_method_arr = []
        return dop_method_arr

    @staticmethod
    def operator_map():
        operator_map = {
            "rotation_x": functools.partial(CalibOperator.rotation, aixs=0),
            "rotation_y": functools.partial(CalibOperator.rotation, aixs=1),
            "rotation_z": functools.partial(CalibOperator.rotation, aixs=2),
        }
        return operator_map

    @staticmethod
    def rotation(V2C, severity=1, aixs=None):
        num = [0.5, 1, 2][severity - 1]
        theta_xyz = [0, 0, 0]
        theta_xyz[aixs] = num
        trans_xyz = [0, 0, 0]
        V2C_noise = CalibOperator.extrinsic_mutaion(V2C, theta_xyz, trans_xyz)
        V2C_noise = V2C_noise.flatten()
        return V2C_noise

    # @staticmethod
    # def rotation_transform_all(V2C):
    #     theta_xyz = np.random.uniform(-1, 1, size=(3,))  # [0,3] degree
    #     trans_xyz = np.random.uniform(0, 0.01, size=(3,))
    #     V2C_noise = CalibOperator.extrinsic_mutaion(V2C, theta_xyz, trans_xyz)
    #     V2C_noise = V2C_noise.flatten()
    #     return V2C_noise
    # 
    # @staticmethod
    # def translation_all(V2C):
    #     theta_xyz = [0, 0, 0]
    #     trans_xyz = np.random.uniform(0, 0.01, size=(3,))
    #     V2C_noise = CalibOperator.extrinsic_mutaion(V2C, theta_xyz, trans_xyz)
    #     V2C_noise = V2C_noise.flatten()
    #     return V2C_noise

    @staticmethod
    def extrinsic_mutaion(V2C, theta_xyz, trans_xyz):
        V2C_noise = np.zeros_like(V2C)
        theta_xyz = [theta * math.pi / 180.0 for theta in theta_xyz]
        theta_xyz = np.array(theta_xyz)

        # trans_xyz = [1 + trans for trans in trans_xyz]
        # trans_xyz = np.array(trans_xyz)

        # add rotaion noise
        angles = rotationMatrixToEulerAngles(V2C[:, :3])
        angles += theta_xyz  # add noise
        V2C_rotation_noise = eulerAnglesToRotationMatrix(angles)
        V2C_noise[:, :3] = V2C_rotation_noise

        # add tans noise
        # V2C_trans_noise = V2C[:, 3] * trans_xyz
        V2C_trans_noise = V2C[:, 3] + trans_xyz
        # print(V2C[:, 3])
        # print(V2C_trans_noise)
        V2C_noise[:, 3] = V2C_trans_noise
        # print(V2C_noise)
        return V2C_noise


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


# # https://blog.csdn.net/ZHUO__zhuo/article/details/124634228
# https://learnopencv.com/rotation-matrix-to-euler-angles/
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
