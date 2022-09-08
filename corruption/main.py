''''
@Project: AI-MSF-benchmark
@Description: Please add Description       
@Time:2022/9/6 16:00       
@Author:NianGao    
 
'''
import os

import fire

'''
input_dir: input  directory 
output_dir: input  directory 
dop : corruption full name
sev: severity level (1-5)
conflict_mode: if file exists generate (0) or skip (1)
copy_mode:  symlink(0) or copy files(1)
fm: calib file format e.g. 000001.txt
'''


def simulate_image(input_dir, output_dir, corruption, sev, conflict_mode=0):
    from corruption.simulation.image_simulation import ImageSimulation
    os.makedirs(output_dir, exist_ok=True)
    ImageSimulation.simulation(input_dir, output_dir, corruption, sev, conflict_mode=conflict_mode)


def simulate_delay(input_dir, output_dir, corruption, sev, copy_mode=0):
    from corruption.simulation.delay_simulation import DelaySimulation
    DelaySimulation.simulation(input_dir, output_dir, corruption, sev, copy_mode=copy_mode)


def simulate_lidar(input_dir, output_dir, corruption, sev, conflict_mode=0):
    from corruption.simulation.lidar_simulation import LidarSimulation
    os.makedirs(output_dir, exist_ok=True)
    LidarSimulation.simulation(input_dir, output_dir, corruption, sev, conflict_mode=conflict_mode)


def simulate_calib(input_dir, output_dir, corruption, sev, fm="%06d.txt", for_depth=False):
    from corruption.simulation.calib_simulation import CalibSimulation
    os.makedirs(output_dir, exist_ok=True)
    if for_depth:
        CalibSimulation.simulation4depth(input_dir, output_dir, corruption, sev)
    else:
        CalibSimulation.simulation(input_dir, output_dir, corruption, sev, fm=fm)


def simulate_weather(input_dir_c, output_dir_c, input_dir_l, output_dir_l, calibdir, depthdir,
                     corruption, sev,
                     for_depth=False,
                     for_tracking=False,
                     x2object_dir=None
                     ):
    from corruption.simulation.weather_simulation import WeatherSimulation
    os.makedirs(output_dir_c, exist_ok=True)
    os.makedirs(output_dir_l, exist_ok=True)
    # if for_depth:
    #     WeatherSimulation.simulation4depth(input_dir, output_dir, corruption, sev)
    # if for_tracking:
    #     WeatherSimulation.simulation4tracking(input_dir, output_dir, corruption, sev)
    # else:
    input_dir_l = os.path.abspath(input_dir_l)
    input_dir_c = os.path.abspath(input_dir_c)
    output_dir_l = os.path.abspath(output_dir_l)
    output_dir_c = os.path.abspath(output_dir_c)
    if for_tracking:
        WeatherSimulation.simulation4tracking(x2object_dir, output_dir_c, input_dir_l, output_dir_l,
                                              depthdir, corruption, sev,
                                              conflict_mode=0)
    elif for_depth:
        WeatherSimulation.simulation4depth(x2object_dir, output_dir_c, output_dir_l, depthdir, corruption, sev,
                                           conflict_mode=0)
    else:
        WeatherSimulation.simulation(input_dir_c, output_dir_c, input_dir_l, output_dir_l,
                                     calibdir, depthdir, corruption, sev)


if __name__ == '__main__':
    fire.Fire()
