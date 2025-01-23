import math
import torch
import numpy as np
from typing import Optional

import wandb


import omni
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.utils.domain_randomization.randomize import Randomizer


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
        self.torque_buffer = torch.zeros(10, self._num_envs, 3, device=self._device)
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

        euler_angles = quaternion_to_euler(self.root_quats)
        euler_rates = compute_euler_rates(euler_angles, angular_velocities)

        eulerz, eulerx, eulery = euler_angles.unbind(dim=-1)

        roll_vel = self.dof_vel[:, self._roll_dof_index]
        pitch_vel = self.dof_vel[:, self._pitch_dof_index]
        yaw_vel = self.dof_vel[:, self._yaw_dof_index]

        self.obs_buf[:, 0] = roll_vel
        self.obs_buf[:, 1] = pitch_vel
        self.obs_buf[:, 2] = yaw_vel
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

        if self._log_wandb:
            wandb.log({
                "Roll Ang Vel": torch.mean(roll_vel).cpu().detach().numpy(),
                "Pitch Ang Vel": torch.mean(pitch_vel).cpu().detach().numpy(),
                "Yaw Ang Vel": torch.mean(yaw_vel).cpu().detach().numpy(),
            })

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
        try:
            t_roll = self._stall_torque - self._stall_torque * torch.abs(self.dof_vel[:, self._roll_dof_index])
            t_pitch = self._stall_torque - self._stall_torque * torch.abs(self.dof_vel[:, self._pitch_dof_index])
        except:
            t_roll = self._max_effort
            t_pitch = self._max_effort

        forces[:, self._roll_dof_index] = torch.clamp(
            t_roll * actions[:, 0], -self._max_effort, self._max_effort
        )
        forces[:, self._pitch_dof_index] = torch.clamp(
            t_pitch * actions[:, 1], -self._max_effort, self._max_effort
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

        self.torque_buffer = torch.roll(self.torque_buffer, -1, dims=0)
        self.torque_buffer[-1] = forces

        if self._log_wandb:
            wandb.log({
                "Roll Torque": torch.mean(forces[:, self._roll_dof_index]).cpu().detach().numpy(),
                "Pitch Torque": torch.mean(forces[:, self._pitch_dof_index]).cpu().detach().numpy(),
                "Yaw Torque": torch.mean(forces[:, self._yaw_dof_index]).cpu().detach().numpy(),
            })

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
        self.torque_buffer[:, env_ids, :] = torch.zeros((10, num_resets, 1), device=self._device)

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
        up_reward = torch.where(self.orient_z >= 0.7, 1.0, 0)
        angle_reward = ups[..., 2]
        fallen_pen = torch.where(self.orient_z <= 0.25, -2, 0)
        # effort = torch.square(torch.mean(self.torque_buffer, dim=0)).sum(-1)
        # effort = torch.abs(torch.mean(self.torque_buffer, dim=0))
        # effort_reward = torch.exp(-3.0 * effort[:, self._roll_dof_index]**2)
        # torque_term = 0.5 * torch.squeeze(torch.abs(torch.mean(self.torque_buffer, dim=0)), dim=1)
        # vel_term = (2 * (0.01 * self.root_vel)**2).sum(-1)
        vel_term_roll = 0.1 * (self.dof_vel[:, self._roll_dof_index]/60)**2
        vel_term_pitch = 0.1 * (self.dof_vel[:, self._pitch_dof_index]/60)**2

        effort_penalty = torch.square(self.torque_buffer[-1, :, self._roll_dof_index])/4
        effort_var_pen = (torch.abs(torch.var(self.torque_buffer, dim=0))/4)

        dist_from_spawn = torch.sqrt(
            torch.square(self.initial_root_pos.clone() - self.root_pos).sum(-1)
        )
        pos_reward = 1.0 - dist_from_spawn**2

        if self._log_wandb:
            wandb.log({
                "Effort Penalty": torch.mean(effort_penalty).cpu().detach().numpy(),
                "Effort Variance Penalty": torch.mean(effort_var_pen).cpu().detach().numpy(),
                "Angle Reward": torch.mean(angle_reward).cpu().detach().numpy(),
                "Velocity Penalty Roll": torch.mean(vel_term_roll).cpu().detach().numpy(),
                "Velocity Penalty Pitch": torch.mean(vel_term_roll).cpu().detach().numpy(),
                "Position Reward": torch.mean(pos_reward).cpu().detach().numpy(),
            })


        self.rew_buf[:] = (
            up_reward + 
            fallen_pen + 
            angle_reward - 
            effort_penalty - 
            vel_term_roll - 
            effort_var_pen[:, self._roll_dof_index] + 
            pos_reward - 
            vel_term_pitch
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



def quaternion_to_euler(quaternions, convention='zyx'):
    w, x, y, z = quaternions.unbind(dim=-1)

    # ZYX convention
    if convention == 'zyx':
        # Yaw (z-axis rotation)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (z ** 2 + x ** 2))

        # Pitch (y-axis rotation)
        sin_pitch = 2 * (w * y - z * x)
        sin_pitch = torch.clamp(sin_pitch, -1.0, 1.0)  # Clamp to avoid NaNs
        pitch = torch.asin(sin_pitch)

        # Roll (x-axis rotation)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))

        return torch.stack((yaw, pitch, roll), dim=-1)
    
def compute_euler_rates(euler_angles, angular_velocity):
    z, x, y = euler_angles.unbind(dim=-1)  # Yaw (z), Pitch (x), Roll (y)

    # Create transformation matrices for ZYX convention (batched)
    sin_x, cos_x = torch.sin(x), torch.cos(x)
    sin_y, cos_y = torch.sin(y), torch.cos(y)

    euler_rate_matrices = torch.stack([
        torch.stack([cos_y, torch.zeros_like(y), sin_y], dim=-1),
        torch.stack([sin_y * torch.tan(x), torch.ones_like(x), -cos_y * torch.tan(x)], dim=-1),
        torch.stack([-sin_y / cos_x, torch.zeros_like(x), cos_y / cos_x], dim=-1)
    ], dim=-2)  # Shape: (N, 3, 3)

    # Batch matrix multiplication
    euler_rates = torch.einsum('bij,bj->bi', euler_rate_matrices, angular_velocity)
    return euler_rates
