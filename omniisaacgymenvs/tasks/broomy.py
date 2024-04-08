import math
import torch
import numpy as np
from typing import Optional
# import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as ort

import omni
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.utils.domain_randomization.randomize import Randomizer

import wandb

EPS = 1e-6

class Broomy(Robot):
    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        name: Optional[str] = "RWIP",
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

        self._num_observations = 14
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

        self.dt = self._task_cfg["sim"]["dt"]

        self.randomize = self._task_cfg['domain_randomization']['randomize']        
        print("ADD RANDOMIZATION? ", self.randomize)

        self._log_wandb = self._cfg['wandb_activate']

    def set_up_scene(self, scene) -> None:
        self.get_broomy()
        super().set_up_scene(scene)
        self._broomys = ArticulationView(
            prim_paths_expr="/World/envs/.*/RWIP/RWIP_SIM_TEST", name="rwip_view", reset_xform_properties=False
        )
        scene.add(self._broomys)
        self.torque_buffer = torch.zeros(10, self._num_envs, 1, device=self._device)
        return

    def get_broomy(self):
        rwip = Broomy(prim_path=self.default_zero_env_path + "/RWIP", usd_path="/home/fizzer/Documents/unicycle_08/RWIP_SIM_TEST_v5.usd", name="RWIP")
        self._sim_config.apply_articulation_settings(
            "RWIP", get_prim_at_path(self.default_zero_env_path + "/RWIP"+"/RWIP_SIM_TEST"), self._sim_config.parse_actor_config("RWIP")
        )

    def get_observations(self) -> dict:
        self.root_pos, self.root_quats = self._broomys.get_world_poses(clone=False)
        dof_vel = self._broomys.get_joint_velocities(clone=False)
        root_velocities = self._broomys.get_velocities(clone=False)

        roll_vel = dof_vel[:, self._roll_dof_index]
        pitch_vel = dof_vel[:, self._pitch_dof_index]
        posns_from_start = root_pos - self._env_pos

        self.obs_buf[:, 0] = roll_vel
        self.obs_buf[:, 1] = pitch_vel
        self.obs_buf[..., 2:5] = posns_from_start
        self.obs_buf[..., 5:9] = root_quats
        self.obs_buf[..., 9:15] = root_velocities

        if self.randomize:
            _observations_uncorrelated_noise = torch.normal(
                    mean=0,
                    std=0.001,
                    size=(self._num_envs, self._num_observations),
                    device=self._cfg["rl_device"],
                )
            self.obs_buf += self._observations_correlated_noise
            self.obs_buf += _observations_uncorrelated_noise
        
        observations = {self._rwips.name: {"obs_buf": self.obs_buf}}
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
        forces = torch.zeros((self._broomys.count, self._num_actions), dtype=torch.float32, device=self._device)

        forces[:, self._roll_dof_index] = torch.clamp(self._max_effort * actions[:, 0], -self._max_effort, self._max_effort)
        forces[:, self._pitch_dof_index] = torch.clamp(self._max_effort * actions[:, 1], -self._max_effort, self._max_effort)

        if self.randomize:
            forces[:, self._roll_dof_index] += self._actions_correlated_noise.squeeze(1)
            forces[:, self._pitch_dof_index] += self._actions_correlated_noise.squeeze(1)

        indices = torch.arange(self._broomys.count, dtype=torch.int32, device=self._device)
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

        self._broomys.set_world_poses(self.initial_root_pos[env_ids].clone(), self.initial_root_rot[env_ids].clone(), indices=env_ids)
        self._broomys.set_velocities(root_velocities[env_ids], indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self) -> None:
        print("DOF Names: ", self._broomys.dof_names)
        self._roll_dof_index = self._broomys.get_dof_index("dof_roll")
        self._pitch_dof_index = self._broomys.get_dof_index("dof_pitch")

        # Save for comoputing reset posn later
        root_pos, root_rot = self._broomys.get_world_poses(clone=False)
        self.initial_root_pos, self.initial_root_rot = root_pos.clone(), root_rot.clone()

        # randomize all envs
        indices = torch.arange(self._broomys.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:   
        # uprightness
        root_quats = self.root_quats
        ups = quat_axis(root_quats, 2)
        self.orient_z = ups[..., 2]
        up_reward = torch.clamp(ups[..., 2], min=0.0, max=1.0)

        effort = torch.square(self.actions).sum(-1)
        effort_reward = 0.05 * torch.exp(-0.5 * effort)

        self.rew_buf[:] = up_reward + effort_reward

    def is_done(self) -> None:
        resets = torch.where(self.orient_z < 0.0, 1, 0)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets