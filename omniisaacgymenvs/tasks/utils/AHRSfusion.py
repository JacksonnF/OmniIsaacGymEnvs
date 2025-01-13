import logging

import imufusion
import numpy as np
import yaml
from transforms3d import euler, quaternions


class AHRSfusion:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.transformation_matrix = np.identity(3)

        # Instantiate sensor fusion algo
        self.offset = imufusion.Offset(self.sample_rate)
        self.ahrs = imufusion.Ahrs()
        self.ahrs.settings = imufusion.Settings(
            imufusion.CONVENTION_NWU,
            0.5,  # 0.5 gain (on the accel error correction)
            500,  # gyroscope range (is this correct)
            1,  # acceleration rejection
            10,  # magnetic rejection
            5 * self.sample_rate,
        )

    def get_next_state(self, accel_data, gyro_data, delta_time):
        mag_data = np.array([0, 0, 0])
        corrected_gyro = self.offset.update(gyro_data)
        self.ahrs.update(
            corrected_gyro,
            accel_data,
            mag_data,
            delta_time,
        )

        euler_angles, euler_rates = self.update_state_from_quaternion(gyro_data)

        return (euler_angles, euler_rates)

    def update_state_from_quaternion(self, gyro_data):
        """Compute and retururn ZYX Euler angles and rotation matrix A q_dot = gyro"""
        angles = euler.quat2euler(self.ahrs.quaternion.wxyz, axes="rzxy")
        euler_angles = np.rad2deg(angles)

        z, x, y = angles
        mat = np.array(
            [
                [np.cos(y), 0, np.sin(y)],
                [1.0 * np.sin(y) * np.tan(x), 1, -1.0 * np.cos(y) * np.tan(x)],
                [-np.sin(y) / np.cos(x), 0, np.cos(y) / np.cos(x)],
            ]
        )
        euler_rates = mat @ (gyro_data * np.pi / 180)

        return (euler_angles, euler_rates)
