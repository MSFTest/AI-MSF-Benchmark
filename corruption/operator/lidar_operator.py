''''
@Project: fusion
@Description: Please add Description       
@Time:2022/9/6 13:55       
@Author:NianGao    
 
'''
import functools
import numpy as np


class LidarOperator(object):
    """
       :Author:  NianGao
       :Create:  2022/1/13
    """

    @staticmethod
    def operator_map():
        operator_map = {
            "loss_partial": LidarOperator.loss_partial,
            "gaussian_noise": LidarOperator.gaussian_noise,
            "impulse_noise": LidarOperator.impulse_noise,
            "loss_complete": LidarOperator.loss_complete,
        }
        return operator_map

    @staticmethod
    def loss_partial(pointcloud, severity=1):
        params = [0.1, 0.25, 0.5, 0.75, 0.9][severity - 1]
        N, C = pointcloud.shape
        index = np.random.permutation(N)[:int(N * params)]
        pointcloud[index] = [0, 0, 0, 0]
        return pointcloud

    @staticmethod
    def gaussian_noise(pointcloud, severity=1):
        N, C = pointcloud.shape
        c = [0.02, 0.04, 0.06, 0.08, 0.1][severity - 1]
        jitter = np.random.normal(size=(N, C)) * c
        new_pc = (pointcloud + jitter).astype('float32')
        return new_pc

    @staticmethod
    def impulse_noise(pointcloud, severity=1):
        N, C = pointcloud.shape
        c = [N // 30, N // 20, N // 10, N // 7, N // 5][severity - 1]
        index = np.random.choice(N, c, replace=False)
        pointcloud[index] += np.random.choice([-1, 1], size=(c, C)) * 0.2
        return pointcloud

    @staticmethod
    def loss_complete(pointcloud, severity=1):
        N, C = pointcloud.shape
        index = range(N)
        pointcloud[index] = [0, 0, 0, 0]
        return pointcloud
