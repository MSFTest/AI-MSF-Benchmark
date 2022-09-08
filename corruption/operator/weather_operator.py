''''
@Project: AI-MSF-benchmark
@Description: Please add Description       
@Time:2022/9/6 15:18       
@Author:NianGao    
 
'''

import os


class WeatherOperator(object):
    @staticmethod
    def operator_map():
        operator_map = {
            "rain": WeatherOperator.rain,
            "fog": WeatherOperator.fog,
        }
        return operator_map

    @staticmethod
    def rain(inputdir_c, ouputdir_c, inputdir_l, outputdir_l, severity, tp=("c", "l"), conflict_mode=0):
        rainfall = [10, 25, 50, 75, 100][severity - 1]
        if "c" in tp:
            print("simulate rain for camera :{}".format(rainfall))
            os.makedirs(ouputdir_c, exist_ok=True)
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "3rd_parts", "camera")
            cmd1 = "cd {}".format(path)
            if conflict_mode == 0:
                cmd2 = "python main.py --dataset kitti --output {} --intensity {}" \
                    .format(ouputdir_c, rainfall)
            else:
                cmd2 = "python main.py --dataset kitti --output {} --intensity {} --conflict_strategy skip" \
                    .format(ouputdir_c, rainfall)
            os.system("{} && {}".format(cmd1, cmd2))
        if "l" in tp:
            print("simulate rain for Lidar :{}".format(rainfall))
            # lidar
            os.makedirs(outputdir_l, exist_ok=True)
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "3rd_parts", "lidar")
            cmd1 = "cd {}".format(path)
            cmd2 = "python main.py  rain --output {} --input {} --intensity {} " \
                .format(outputdir_l, inputdir_l, rainfall)
            os.system("{} && {}".format(cmd1, cmd2))

    @staticmethod
    def fog(camerainput, cameraoutput, lidarinput, lidaroutput, severity, tp=("c", "l"), conflict_mode=0):
        assert severity <= 3
        level = ["strong", "moderate", "low"][severity - 1]
        if "c" in tp:
            # camera
            print("simulate fog for camera")
            os.makedirs(cameraoutput, exist_ok=True)
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "3rd_parts", "camera")
            cmd1 = "cd {}".format(path)
            cmd2 = "python fog.py  --output {} --level {} --conflict_mode {}" \
                .format(cameraoutput, level, conflict_mode)
            os.system("{} && {}".format(cmd1, cmd2))
        if "l" in tp:
            print("simulate fog for Lidar")
            # lidar
            os.makedirs(lidaroutput, exist_ok=True)
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "3rd_parts", "lidar")
            cmd1 = "cd {}".format(path)
            cmd2 = "python main.py  fog --output {} --input {} --level {} --conflict_mode {}" \
                .format(lidaroutput, lidarinput, level, conflict_mode)
            os.system("{} && {}".format(cmd1, cmd2))
