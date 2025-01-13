import math
import torch
import numpy as np
from typing import Optional

# import pandas as pd
import matplotlib.pyplot as plt
import threading
from transforms3d import euler, quaternions
import imufusion

import omni
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.sensor import IMUSensor

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.utils.domain_randomization.randomize import Randomizer
from omniisaacgymenvs.tasks.utils.AHRSfusion import AHRSfusion


EPS = 1e-6


class Broomy(Robot):
    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        name: Optional[str] = "BROOMY",
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )


class BroomyTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)
        self._max_episode_length = 350

        self._num_observations = 9
        self._num_actions = 3
        RLTask.__init__(self, name, env)
        if self.randomize:
            self._observations_correlated_noise = torch.normal(
                mean=0,
                std=0.01,
                size=(self._num_envs, self._num_observations),
                device=self._cfg["rl_device"],
            )
            self._actions_correlated_noise = torch.normal(
                mean=0,
                std=0.001,
                size=(self._num_envs, self._num_actions),
                device=self._cfg["rl_device"],
            )
        self.ahrs_insts = [
            AHRSfusion(int(1 / 0.01)) for i in range(self._num_envs)
        ]  # TODO: Config this
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 1.0])

        self._max_effort = self._task_cfg["env"]["maxEffort"]
        self._max_effort_yaw = self._task_cfg["env"]["maxEffortYaw"]
        self._stall_torque = self._task_cfg["env"]["stallTorque"]

        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]

        self.dt = self._task_cfg["sim"]["dt"]

        self.randomize = self._task_cfg["domain_randomization"]["randomize"]
        print("ADD RANDOMIZATION? ", self.randomize)

        self._log_wandb = self._cfg["wandb_activate"]

    def set_up_scene(self, scene) -> None:
        self.get_broomy()
        super().set_up_scene(scene)
        self._broomys = ArticulationView(
            prim_paths_expr="/World/envs/.*/Broomy/full_robot",
            name="broomy_view",
            reset_xform_properties=False,
        )
        scene.add(self._broomys)
        self.imus = self.create_sensors()
        self.torque_buffer = torch.zeros(10, self._num_envs, 1, device=self._device)
        return

    def create_sensors(self) -> list:
        sensor_paths = [
            f"/World/envs/env_{i}/Broomy/full_robot/robot_body/Imu{i}"
            for i in range(self._num_envs)
        ]
        sensors = []
        for path in sensor_paths:
            imu = IMUSensor(
                prim_path=path,
                name="imu",
                # frequency=60,
                dt=0.005,  # same as config (can set to that var)
                translation=np.array([0, 0, 0]),
                orientation=np.array([0, 0, 0, 1]),
                linear_acceleration_filter_size=10,
                angular_velocity_filter_size=10,
                orientation_filter_size=10,
            )
            sensors.append(imu)
        return sensors

    def get_broomy(self):
        broomy = Broomy(
            prim_path=self.default_zero_env_path + "/Broomy",
            usd_path="/home/fizzer/Documents/unicycle_29/no_banana_broomy.usd",
            name="Broomy",
        )
        self._sim_config.apply_articulation_settings(
            "Broomy",
            get_prim_at_path(self.default_zero_env_path + "/Broomy" + "/full_robot"),
            self._sim_config.parse_actor_config("Broomy"),
        )

    def read_imus(self):
        n = len(self.imus)
        readings = [self.imus[i].get_current_frame() for i in range(n)]
        accel_data = np.array([readings[i]["lin_acc"].cpu().numpy() for i in range(n)])
        gyro_data = np.array([readings[i]["ang_vel"].cpu().numpy() for i in range(n)])

        next_states = torch.tensor(
            np.array(
                [
                    self.ahrs_insts[i].get_next_state(accel_data[i], gyro_data[i], 0.001)
                    for i in range(n)
                ]
            ),
            device=self._device,
        )
        return next_states

    def get_observations(self) -> dict:
        # Get observed quantities
        self.root_pos, self.root_quats = self._broomys.get_world_poses(clone=False)
        dof_vel = self._broomys.get_joint_velocities(clone=False)

        imu_readings = self.read_imus()
        euler_angles = imu_readings[:, 0, :]
        euler_rates = imu_readings[:, 1, :]

        eulerx, eulery, eulerz = (
            euler_angles[:, 0],
            euler_angles[:, 1],
            euler_angles[:, 2],
        )

        self.obs_buf[:, 0] = dof_vel[:, self._roll_dof_index]
        self.obs_buf[:, 1] = dof_vel[:, self._pitch_dof_index]
        self.obs_buf[:, 2] = dof_vel[:, self._yaw_dof_index]
        self.obs_buf[:, 3] = eulerx
        self.obs_buf[:, 4] = eulery
        self.obs_buf[:, 5] = eulerz
        self.obs_buf[:, 6:9] = euler_rates

        if self.randomize:
            _observations_uncorrelated_noise = torch.normal(
                mean=0,
                std=0.001,
                size=(self._num_envs, self._num_observations),
                device=self._cfg["rl_device"],
            )
            self.obs_buf += self._observations_correlated_noise
            self.obs_buf += _observations_uncorrelated_noise

        observations = {self._broomys.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
            if self.randomize:
                self._observations_correlated_noise[reset_env_ids] = torch.normal(
                    mean=0,
                    std=0.01,
                    size=(len(reset_env_ids), self._num_observations),
                    device=self._cfg["rl_device"],
                )
                self._actions_correlated_noise = torch.normal(
                    mean=0,
                    std=0.001,
                    size=(self._num_envs, self._num_actions),
                    device=self._cfg["rl_device"],
                )

        self.actions = actions.to(self._device)
        forces = torch.zeros(
            (self._broomys.count, self._num_actions),
            dtype=torch.float32,
            device=self._device,
        )

        forces[:, self._roll_dof_index] = torch.clamp(
            self._max_effort * actions[:, 0], -self._max_effort, self._max_effort
        )
        forces[:, self._pitch_dof_index] = torch.clamp(
            self._max_effort * actions[:, 1], -self._max_effort, self._max_effort
        )
        forces[:, self._yaw_dof_index] = torch.clamp(
            self._max_effort_yaw * actions[:, 2],
            -self._max_effort_yaw,
            self._max_effort_yaw,
        )

        if self.randomize:
            forces[:, self._roll_dof_index] += self._actions_correlated_noise.squeeze(1)
            forces[:, self._pitch_dof_index] += self._actions_correlated_noise.squeeze(
                1
            )
            forces[:, self._roll_dof_index] += self._actions_correlated_noise.squeeze(1)

        indices = torch.arange(
            self._broomys.count, dtype=torch.int32, device=self._device
        )
        self._broomys.set_joint_efforts(forces, indices=indices)

    def reset_idx(self, env_ids) -> None:
        num_resets = len(env_ids)

        dof_pos = torch.zeros((num_resets, self._broomys.num_dof), device=self._device)
        dof_vel = torch.zeros((num_resets, self._broomys.num_dof), device=self._device)
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0
        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._broomys.set_joint_positions(dof_pos, indices=indices)
        self._broomys.set_joint_velocities(dof_vel, indices=indices)

        self._broomys.set_world_poses(
            self.initial_root_pos[env_ids].clone(),
            self.initial_root_rot[env_ids].clone(),
            indices=env_ids,
        )
        self._broomys.set_velocities(root_velocities[env_ids], indices=env_ids)

        # Reset AHRSfusion
        for ind in indices:
            self.ahrs_insts[ind].ahrs.reset()  # TODO: get correct function

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self) -> None:
        self._roll_dof_index = self._broomys.get_dof_index("roll")
        self._pitch_dof_index = self._broomys.get_dof_index("pitch")
        self._yaw_dof_index = self._broomys.get_dof_index("yaw")

        # Save for comoputing reset posn later
        root_pos, root_rot = self._broomys.get_world_poses(clone=False)
        self.root_velocities = self._broomys.get_velocities(clone=False)
        self.initial_root_pos, self.initial_root_rot = (
            root_pos.clone(),
            root_rot.clone(),
        )

        # randomize all envs
        indices = torch.arange(
            self._broomys.count, dtype=torch.int64, device=self._device
        )
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # uprightness
        root_quats = self.root_quats

        ups = quat_axis(root_quats, 2)
        self.orient_z = ups[..., 2]
        up_reward = torch.where(self.orient_z >= 0.7, 1.0, 0)
        angle_reward = ups[..., 2]
        fallen_pen = torch.where(self.orient_z <= 0.25, -1, 0)
        effort = torch.square(self.actions).sum(-1)
        effort_reward = 0.05 * torch.exp(-0.5 * effort)
        dist_from_spawn = torch.sqrt(
            torch.square(self.initial_root_pos.clone() - self.root_pos).sum(-1)
        )
        pos_reward = 1.0 / (1.0 + 3 * dist_from_spawn * dist_from_spawn)

        self.rew_buf[:] = (
            up_reward + fallen_pen + angle_reward + effort_reward + pos_reward
        )

    def is_done(self) -> None:
        resets = torch.where(self.orient_z < 0.1, 1, 0)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def quats_to_euler_rates(euler_angles, angular_velocities):
    # x, y, z = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    x, y, z = euler_angles
    cos_x = torch.cos(x)
    cos_y = torch.cos(y)
    sin_x = torch.sin(x)
    tan_y = torch.tan(y)

    t11 = torch.ones_like(x)
    t12 = sin_x * tan_y
    t13 = cos_x * tan_y
    t21 = torch.zeros_like(x)
    t22 = cos_x
    t23 = -sin_x
    t31 = torch.zeros_like(x)
    t32 = sin_x / cos_y
    t33 = cos_x / cos_y

    T = torch.stack(
        [
            torch.stack([t11, t12, t13], dim=-1),
            torch.stack([t21, t22, t23], dim=-1),
            torch.stack([t31, t32, t33], dim=-1),
        ],
        dim=-2,
    )

    angular_velocities = angular_velocities.unsqueeze(-1)

    euler_rates = torch.matmul(T, angular_velocities).squeeze(-1)

    return euler_rates
