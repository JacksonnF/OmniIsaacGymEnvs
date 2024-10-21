import math
import torch
import numpy as np
from typing import Optional

# import pandas as pd
import matplotlib.pyplot as plt

import omni
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

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

        self._num_observations = 17
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
            print("INITIAL CORRELATED NOISE: ", self._observations_correlated_noise)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])

        self._max_effort = self._task_cfg["env"]["maxEffort"]
        self._stall_torque = self._task_cfg["env"]["stallTorque"]

        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]

        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"][
            "linear_x"
        ]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"][
            "linear_y"
        ]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"][
            "yaw"
        ]
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"][
            "linearVelocityXYRewardScale"
        ]
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"][
            "angularVelocityZRewardScale"
        ]

        self.dt = self._task_cfg["sim"]["dt"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self.randomize = self._task_cfg["domain_randomization"]["randomize"]
        print("ADD RANDOMIZATION? ", self.randomize)

        self._log_wandb = self._cfg["wandb_activate"]

    def set_up_scene(self, scene) -> None:
        self.get_broomy()
        super().set_up_scene(scene)
        self._broomys = ArticulationView(
            prim_paths_expr="/World/envs/.*/Broomy/top_level_broomy_sim",
            name="broomy_view",
            reset_xform_properties=False,
        )
        scene.add(self._broomys)
        self.torque_buffer = torch.zeros(10, self._num_envs, 1, device=self._device)
        return

    def get_broomy(self):
        broomy = Broomy(
            prim_path=self.default_zero_env_path + "/Broomy",
            usd_path="/home/fizzer/Documents/unicycle_08/top_level_broomy_sim.usd",
            name="Broomy",
        )
        self._sim_config.apply_articulation_settings(
            "Broomy",
            get_prim_at_path(
                self.default_zero_env_path + "/Broomy" + "/top_level_broomy_sim"
            ),
            self._sim_config.parse_actor_config("Broomy"),
        )

    def get_observations(self) -> dict:
        self.root_pos, self.root_quats = self._broomys.get_world_poses(clone=False)
        dof_vel = self._broomys.get_joint_velocities(clone=False)
        self.root_vel = self._broomys.get_velocities(clone=False)

        self.base_lin_vel = quat_rotate_inverse(self.root_quats, self.root_vel[:, 0:3])
        self.base_ang_vel = quat_rotate_inverse(self.root_quats, self.root_vel[:, 3:6])

        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])

        self.commands[:, 2] = torch.clip(
            0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
        )

        roll_vel = dof_vel[:, self._roll_dof_index]
        pitch_vel = dof_vel[:, self._pitch_dof_index]
        posns_from_start = self.root_pos - self._env_pos

        self.obs_buf[:, 0] = roll_vel
        self.obs_buf[:, 1] = pitch_vel
        self.obs_buf[..., 2:5] = posns_from_start
        self.obs_buf[..., 5:9] = self.root_quats
        self.obs_buf[..., 9:15] = self.root_vel
        self.obs_buf[..., 15:18] = self.commands[:, :3] * self.commands_scale

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

        actions = actions.to(self._device)
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

        if self.randomize:
            forces[:, self._roll_dof_index] += self._actions_correlated_noise.squeeze(1)
            forces[:, self._pitch_dof_index] += self._actions_correlated_noise.squeeze(
                1
            )

        indices = torch.arange(
            self._broomys.count, dtype=torch.int32, device=self._device
        )
        self._broomys.set_joint_efforts(forces, indices=indices)

    def reset_idx(self, env_ids) -> None:
        num_resets = len(env_ids)

        dof_pos = torch.zeros((num_resets, self._broomys.num_dof), device=self._device)
        dof_vel = torch.zeros((num_resets, self._broomys.num_dof), device=self._device)
        root_velocities = torch.zeros((num_resets, 6), device=self._device)

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
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_x_range[0],
            self.command_x_range[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_y_range[0],
            self.command_y_range[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze()
        self.commands[env_ids, 3] = torch_rand_float(
            self.command_yaw_range[0],
            self.command_yaw_range[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze()
        self.commands[env_ids] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.25
        ).unsqueeze(
            1
        )  # set small commands to zero
        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self) -> None:
        print("DOF Names: ", self._broomys.dof_names)
        self._roll_dof_index = self._broomys.get_dof_index("Revolute_1")
        self._pitch_dof_index = self._broomys.get_dof_index("Revolute_1_01")

        # Save for comoputing reset posn later
        root_pos, root_rot = self._broomys.get_world_poses(clone=False)
        self.initial_root_pos, self.initial_root_rot = (
            root_pos.clone(),
            root_rot.clone(),
        )

        self.commands = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            device=self.device,
            requires_grad=False,
        )
        self.forward_vec = torch.tensor(
            [1.0, 0.0, 0.0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

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
        up_reward = torch.clamp(ups[..., 2], min=0.0, max=1.0)

        # effort penalty
        effort = torch.square(self.actions).sum(-1)
        effort_reward = 0.05 * torch.exp(-0.5 * effort)

        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_lin_vel_xy = (
            torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        )
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        self.rew_buf[:] = up_reward + effort_reward + rew_lin_vel_xy + rew_ang_vel_z

    def is_done(self) -> None:
        resets = torch.where(self.orient_z < 0.0, 1, 0)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles
