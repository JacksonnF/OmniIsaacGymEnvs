import math
import torch
import numpy as np
from typing import Optional
from collections import namedtuple

import wandb


import omni
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.utils.domain_randomization.randomize import Randomizer

hyperparams = {
    "epsilon": 1e-7, # set to follow what is in the code below 
    "penalty_coeff_roll_vel": 0.085,
    "penalty_coeff_pitch_vel": 0.75, 
    "penalty_coeff_roll_torque": 0.25,
    "penalty_coeff_pitch_torque": 0.25,
    "penalty_coeff_dist_from_spawn": 1.0, # currently, penalty is (1-r^2)
    "penalty_exponent_dist_from_spawn": 1.0,
}


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

        self._num_observations = 8
        self._num_actions = 2
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
            self._randomizer = Randomizer(self._cfg, self._task_cfg)
            print("INITIAL CORRELATED NOISE: ", self._observations_correlated_noise)
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
        self._max_abs_torque_roll = self._task_cfg["env"]["maxAbsTorqueRoll"]
        self._max_abs_torque_pitch = self._task_cfg["env"]["maxAbsTorqueRoll"]
        self._max_abs_torque_yaw = self._task_cfg["env"]["maxAbsTorqueYaw"]
        self._max_abs_motor_speed_roll = self._task_cfg["env"]["maxAbsMotorSpeedRoll"]
        self._max_abs_motor_speed_pitch = self._task_cfg["env"]["maxAbsMotorSpeedPitch"]
        self._max_abs_motor_speed_yaw = self._task_cfg["env"]["maxAbsMotorSpeedYaw"]

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
        if self.randomize:
            self._randomizer.set_up_domain_randomization(self)
            # self._randomizer.randomize_mass_on_startup('broomy_view', 
                                                    #    distribution='uniform', distribution_parameters=[0.1, 0.4], operation='additive')
        self.torque_buffer = torch.zeros(
            10, self._num_envs, self._num_actions, device=self._device
        )
        return

    def get_broomy(self):
        broomy = Broomy(
            prim_path=self.default_zero_env_path + "/Broomy",
            # usd_path="/home/fizzer/Documents/unicycle_29/no_banana_broomy.usd",
            usd_path="/home/fizzer/Documents/broomy-2_11/full_robot_saved.usd",
            name="Broomy",
        )
        self._sim_config.apply_articulation_settings(
            "Broomy",
            get_prim_at_path(self.default_zero_env_path + "/Broomy" + "/full_robot"),
            self._sim_config.parse_actor_config("Broomy"),
        )

    def get_observations(self) -> dict:
        self.root_pos, self.root_quats = self._broomys.get_world_poses(clone=False)
        self.dof_vel = self._broomys.get_joint_velocities(clone=False)
        self.root_vel = self._broomys.get_velocities(clone=False)

        angular_velocities = self.root_vel[:, 3:]

        euler_angles = quaternion_to_euler_zxy(self.root_quats)
        euler_rates = euler_rates_zxy(euler_angles, angular_velocities)

        eulerz, eulerx, eulery = euler_angles.unbind(dim=-1)

        roll_vel = self.dof_vel[:, self._roll_dof_index]
        pitch_vel = self.dof_vel[:, self._pitch_dof_index]
        yaw_vel = self.dof_vel[:, self._yaw_dof_index]


        self.obs_buf[:, 0] = roll_vel
        self.obs_buf[:, 1] = pitch_vel
        self.obs_buf[:, 2] = yaw_vel
        self.obs_buf[:, 3] = eulerx
        self.obs_buf[:, 4] = eulery
        # self.obs_buf[:, 5] = eulerz
        self.obs_buf[:, 5:8] = euler_rates

        if self.randomize:
            _observations_uncorrelated_noise = torch.normal(
                mean=0,
                std=0.001,
                size=(self._num_envs, self._num_observations),
                device=self._cfg["rl_device"],
            )
            self.obs_buf += self._observations_correlated_noise
            self.obs_buf += _observations_uncorrelated_noise

        if self._log_wandb:
            wandb.log(
                {
                    "Roll Ang Vel": torch.mean(roll_vel).cpu().detach().numpy(),
                    "Pitch Ang Vel": torch.mean(pitch_vel).cpu().detach().numpy(),
                    "Yaw Ang Vel": torch.mean(yaw_vel).cpu().detach().numpy(),
                }
            )

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
                # omni.replicator.isaac.physics_view.step_randomization(reset_env_ids)

        self.actions = actions.to(self._device)
        forces = torch.zeros(
            (self._broomys.count, self._num_actions + 1),
            dtype=torch.float32,
            device=self._device,
        )

        max_torque_roll = 0.0
        min_torque_roll = 0.0
        max_torque_pitch = 0.0
        min_torque_pitch = 0.0
        max_torque_yaw = 0.0
        min_torque_yaw = 0.0
        

        try:
            max_torque_roll = self._max_abs_torque_roll if self.dof_vel[:, self._roll_dof_index] < self._max_abs_motor_speed_roll else 0.0
            min_torque_roll = -1.0 * self._max_abs_torque_roll if self.dof_vel[:, self._roll_dof_index] > -self._max_abs_motor_speed_roll else 0.0
            max_torque_pitch = self._max_abs_torque_pitch if self.dof_vel[:, self._pitch_dof_index] < self._max_abs_motor_speed_pitch else 0.0
            min_torque_pitch = -1.0 * self._max_abs_torque_pitch if self.dof_vel[:, self._pitch_dof_index] > -self._max_abs_motor_speed_pitch else 0.0
            max_torque_yaw = self._max_abs_torque_yaw if self.dof_vel[:, self._yaw_dof_index] < self._max_abs_motor_speed_yaw else 0.0
            min_torque_yaw = -1.0 * self._max_abs_torque_yaw if self.dof_vel[:, self._yaw_dof_index] > -self._max_abs_motor_speed_yaw else 0.0

        except:
            max_torque_roll = self._max_abs_torque_roll
            min_torque_roll = -self._max_abs_torque_roll
            max_torque_pitch = self._max_abs_torque_pitch
            min_torque_pitch = -self._max_abs_torque_pitch
            max_torque_yaw = self._max_abs_torque_yaw
            min_torque_yaw = -self._max_abs_torque_yaw

        forces[:, self._roll_dof_index] = torch.clamp(
            max_torque_roll * actions[:, 0], min_torque_roll, max_torque_roll
        )
        forces[:, self._pitch_dof_index] = torch.clamp(
            max_torque_pitch * actions[:, 1],  min_torque_pitch, max_torque_pitch
        )
        # forces[:, self._yaw_dof_index] = torch.clamp(
        #     self._max_effort_yaw * actions[:, 2],
        #     -self._max_effort_yaw,
        #     self._max_effort_yaw,
        # )

        if self.randomize:
            forces[:, self._roll_dof_index] += self._actions_correlated_noise[
                :, self._roll_dof_index
            ]
            forces[:, self._pitch_dof_index] += self._actions_correlated_noise[
                :, self._pitch_dof_index
            ]
            # forces[:, self._yaw_dof_index] += self._actions_correlated_noise[
            #     :, self._yaw_dof_index
            # ]

        self.torque_buffer = torch.roll(self.torque_buffer, -1, dims=0)
        self.torque_buffer[-1] = torch.stack(
            (forces[:, self._roll_dof_index], forces[:, self._roll_dof_index]), dim=1
        )

        if self._log_wandb:
            wandb.log(
                {
                    "Roll Torque": torch.mean(forces[:, self._roll_dof_index])
                    .cpu()
                    .detach()
                    .numpy(),
                    "Pitch Torque": torch.mean(forces[:, self._pitch_dof_index])
                    .cpu()
                    .detach()
                    .numpy(),
                    # "Yaw Torque": torch.mean(forces[:, self._yaw_dof_index])
                    # .cpu()
                    # .detach()
                    # .numpy(),
                }
            )

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

        max_angle = torch.tensor(15.0 * torch.pi / 180.0, device=self._device)
        euler_angles = torch.zeros((num_resets, 3), device=self._device)
        euler_angles[:, 0] = (
            torch.rand(num_resets, device=self._device) * 2 * max_angle - max_angle
        )  # roll
        euler_angles[:, 1] = (
            torch.rand(num_resets, device=self._device) * 2 * max_angle - max_angle
        )  # pitch
        euler_angles[:, 2] = 0.0

        root_rot_rand_euler = euler_angles_to_quats(euler_angles)

        self._broomys.set_world_poses(
            self.initial_root_pos[env_ids].clone(),
            # self.initial_root_rot[env_ids].clone(),
            root_rot_rand_euler,
            indices=env_ids,
        )
        self._broomys.set_velocities(root_velocities[env_ids], indices=env_ids)
        self.torque_buffer[:, env_ids, :] = torch.zeros(
            (10, num_resets, 1), device=self._device
        )

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self) -> None:
        print("DOF Names: ", self._broomys.dof_names)
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
        root_quats = self.root_quats

        ups = quat_axis(root_quats, 2)
        self.orient_z = ups[..., 2]
        up_reward = torch.where(self.orient_z >= 0.85, 1.0, 0)
        angle_reward = ups[..., 2] * 2
        fallen_pen = torch.where(self.orient_z <= 0.5, -5, 0)

        vel_term_roll = hyperparams["penalty_coeff_roll_vel"] * (self.dof_vel[:, self._roll_dof_index] / 60) ** 4
        vel_term_pitch = hyperparams["penalty_coeff_pitch_vel"] * (self.dof_vel[:, self._pitch_dof_index] / 60) ** 4

        effort_penalty_roll = (
            hyperparams["penalty_coeff_roll_torque"] * torch.mean(torch.abs(self.torque_buffer[:, :, self._roll_dof_index]), dim=0)
        )
        effort_penalty_pitch = (
            hyperparams["penalty_coeff_pitch_torque"] * torch.mean(torch.abs(self.torque_buffer[:, :, self._pitch_dof_index]), dim=0)
        )

        # effort_variance = torch.abs(torch.var(self.torque_buffer, dim=0)) / 4

        dist_from_spawn = torch.sqrt(
            torch.square(self.initial_root_pos.clone() - self.root_pos).sum(-1)
        )
        pos_reward = (1.0 - (hyperparams["penalty_coeff_dist_from_spawn"] * dist_from_spawn)) ** hyperparams["penalty_exponent_dist_from_spawn"]

        if self._log_wandb:
            wandb.log(
                {
                    "Effort Penalty": torch.mean(effort_penalty_roll)
                    .cpu()
                    .detach()
                    .numpy(),
                    # "Effort Variance Penalty": torch.mean(effort_var_pen)
                    # .cpu()
                    # .detach()
                    # .numpy(),
                    "Angle Reward": torch.mean(angle_reward).cpu().detach().numpy(),
                    "Velocity Penalty Roll": torch.mean(vel_term_roll)
                    .cpu()
                    .detach()
                    .numpy(),
                    "Velocity Penalty Pitch": torch.mean(vel_term_pitch)
                    .cpu()
                    .detach()
                    .numpy(),
                    "Position Reward": torch.mean(pos_reward).cpu().detach().numpy(),
                }
            )

        self.rew_buf[:] = (
            up_reward
            + fallen_pen
            + angle_reward
            - effort_penalty_roll
            - effort_penalty_pitch
            - vel_term_roll
            + pos_reward
            - vel_term_pitch
            # - effort_variance[:, self._roll_dof_index]
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


def quaternion_to_euler(quaternions, convention="zyx"):
    w, x, y, z = quaternions.unbind(dim=-1)

    # ZYX convention
    if convention == "zyx":
        # Yaw (z-axis rotation)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (z**2 + x**2))

        # Pitch (y-axis rotation)
        sin_pitch = 2 * (w * y - z * x)
        sin_pitch = torch.clamp(sin_pitch, -1.0, 1.0)  # Clamp to avoid NaNs
        pitch = torch.asin(sin_pitch)

        # Roll (x-axis rotation)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

        return torch.stack((yaw, pitch, roll), dim=-1)


def compute_euler_rates(euler_angles, angular_velocity):
    z, x, y = euler_angles.unbind(dim=-1)  # Yaw (z), Pitch (x), Roll (y)

    # Create transformation matrices for ZYX convention (batched)
    sin_x, cos_x = torch.sin(x), torch.cos(x)
    sin_y, cos_y = torch.sin(y), torch.cos(y)

    euler_rate_matrices = torch.stack(
        [
            torch.stack([cos_y, torch.zeros_like(y), sin_y], dim=-1),
            torch.stack(
                [sin_y * torch.tan(x), torch.ones_like(x), -cos_y * torch.tan(x)],
                dim=-1,
            ),
            torch.stack([-sin_y / cos_x, torch.zeros_like(x), cos_y / cos_x], dim=-1),
        ],
        dim=-2,
    )  # Shape: (N, 3, 3)

    # Batch matrix multiplication
    euler_rates = torch.einsum("bij,bj->bi", euler_rate_matrices, angular_velocity)
    return euler_rates


def quaternion_to_euler_zxy(quaternions):
    w, x, y, z = quaternions.unbind(dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    # Build the rotation matrix
    rotation_matrix = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).view(-1, 3, 3)

    # Extract elements for ZXY Euler angles conversion
    # Pitch (X) is arcsin of -R[:, 1, 2]
    pitch = torch.asin(-rotation_matrix[:, 1, 2])

    # Compute cosine of pitch
    cos_pitch = torch.cos(pitch)

    # Threshold to handle gimbal lock
    epsilon = hyperparams['epsilon']
    safe_cos_pitch = torch.where(cos_pitch.abs() < epsilon, epsilon, cos_pitch)

    # Mask for non-gimbal lock cases
    mask = cos_pitch.abs() >= epsilon

    # Initialize angles
    yaw = torch.zeros_like(pitch)
    roll = torch.zeros_like(pitch)

    # Compute yaw and roll when not in gimbal lock
    yaw_valid = torch.atan2(
        rotation_matrix[:, 1, 0] / safe_cos_pitch,
        rotation_matrix[:, 1, 1] / safe_cos_pitch,
    )
    roll_valid = torch.atan2(
        rotation_matrix[:, 0, 2] / safe_cos_pitch,
        rotation_matrix[:, 2, 2] / safe_cos_pitch,
    )

    # Compute yaw and roll in gimbal lock (cos_pitch ~ 0)
    yaw_gimbal = torch.atan2(rotation_matrix[:, 0, 1], rotation_matrix[:, 0, 0])
    roll_gimbal = torch.zeros_like(pitch)

    # Apply mask to select valid or gimbal case
    yaw = torch.where(mask, yaw_valid, yaw_gimbal)
    roll = torch.where(mask, roll_valid, roll_gimbal)

    # Stack angles into (yaw_z, pitch_x, roll_y)
    euler_angles = torch.stack((yaw, pitch, roll), dim=-1)

    return euler_angles


def euler_rates_zxy(euler_angles, angular_velocities, epsilon=1e-7):
    """
    Compute Euler angle rates from Euler angles and angular velocities for ZXY convention.

    Args:
        euler_angles (torch.Tensor): Tensor of shape (N, 3) in (yaw_z, pitch_x, roll_y) order [radians].
        angular_velocities (torch.Tensor): Tensor of shape (N, 3) in (omega_x, omega_y, omega_z) order [radians/sec].
        epsilon (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Euler angle rates in radians/sec as tensor of shape (N, 3) in (yaw_dot, pitch_dot, roll_dot) order.
    """
    # Split Euler angles into yaw (z), pitch (x), roll (y)
    yaw_z, pitch_x, roll_y = euler_angles.unbind(dim=-1)

    # Split angular velocities into omega_x, omega_y, omega_z
    omega_x, omega_y, omega_z = angular_velocities.unbind(dim=-1)

    # Compute trigonometric terms for roll (phi) and pitch (theta)
    sin_phi = torch.sin(roll_y)
    cos_phi = torch.cos(roll_y)
    sin_theta = torch.sin(pitch_x)
    cos_theta = torch.cos(pitch_x)

    # Avoid division by zero by adding epsilon to cos_theta
    cos_theta_safe = cos_theta + epsilon

    # Compute yaw rate (dψ/dt)
    yaw_dot = (-sin_phi * omega_x + cos_phi * omega_z) / cos_theta_safe

    # Compute pitch rate (dθ/dt)
    pitch_dot = cos_phi * omega_x + sin_phi * omega_z

    # Compute roll rate (dφ/dt)
    roll_dot = omega_y + (sin_phi * omega_x - cos_phi * omega_z) * (
        sin_theta / cos_theta_safe
    )

    # Stack the rates into the correct order (yaw_dot, pitch_dot, roll_dot)
    euler_rates = torch.stack((yaw_dot, pitch_dot, roll_dot), dim=-1)

    return euler_rates
