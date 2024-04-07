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

class RWIP(Robot):
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
    
class RWIPTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)
        self._max_episode_length = 350

        self._num_observations = 3
        self._num_actions = 1
        RLTask.__init__(self, name, env)
        if self.randomize: 
            # self._observations_correlated_noise = torch.zeros(
            # (self._num_envs, self._num_observations), device=self._cfg["rl_device"]
            # )
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
        self.get_rwip()
        super().set_up_scene(scene)
        self._rwips = ArticulationView(
            prim_paths_expr="/World/envs/.*/RWIP/RWIP_SIM_TEST", name="rwip_view", reset_xform_properties=False
        )
        scene.add(self._rwips)
        self.torque_buffer = torch.zeros(10, self._num_envs, 1, device=self._device)
        return

    def get_rwip(self):
        rwip = RWIP(prim_path=self.default_zero_env_path + "/RWIP", usd_path="/home/fizzer/Documents/unicycle_08/RWIP_SIM_TEST_v5.usd", name="RWIP")
        self._sim_config.apply_articulation_settings(
            "RWIP", get_prim_at_path(self.default_zero_env_path + "/RWIP"+"/RWIP_SIM_TEST"), self._sim_config.parse_actor_config("RWIP")
        )

    def get_observations(self) -> dict:
        dof_pos = self._rwips.get_joint_positions(clone=False)
        dof_vel = self._rwips.get_joint_velocities(clone=False)

        self.rxnwheel_vel = dof_vel[:, self._rxnwheel_dof_idx]
        self.axis_pos = dof_pos[:, self._axis_dof_idx]
        self.axis_vel = dof_vel[:, self._axis_dof_idx]

        self.obs_buf[:, 0] = self.rxnwheel_vel
        self.obs_buf[:, 1] = self.axis_pos
        self.obs_buf[:, 2] = self.axis_vel

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
        forces = torch.zeros((self._rwips.count, self._rwips.num_dof), dtype=torch.float32, device=self._device)

        try:
            t = self._stall_torque - self._stall_torque * torch.abs(self.rxnwheel_vel) / (45 * 2 * np.pi)
        except:
            t = self._max_effort

        forces[:, self._rxnwheel_dof_idx] = torch.clamp(t * actions[:, 0], -self._max_effort, self._max_effort)
        if self.randomize:
            forces[:, self._rxnwheel_dof_idx] += self._actions_correlated_noise.squeeze(1)
        
        self.torque_buffer = torch.roll(self.torque_buffer, -1, dims=0)
        self.torque_buffer[-1] = forces[:, self._rxnwheel_dof_idx].unsqueeze(-1)

        indices = torch.arange(self._rwips.count, dtype=torch.int32, device=self._device)
        self._rwips.set_joint_efforts(forces, indices=indices)


    def reset_idx(self, env_ids) -> None:
        num_resets = len(env_ids)

        # randomize pendulumn axis position
        dof_pos = torch.zeros((num_resets, self._rwips.num_dof), device=self._device)
        dof_pos[:, self._axis_dof_idx] = 0.23 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._rwips.num_dof), device=self._device)
        dof_vel[:, self._axis_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._rxnwheel_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._rwips.set_joint_positions(dof_pos, indices=indices)
        self._rwips.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self) -> None:
        print(self._rwips.dof_names)
        self._rxnwheel_dof_idx = self._rwips.get_dof_index("dof_rxnwheel")
        self._axis_dof_idx = self._rwips.get_dof_index("dof_axis")

        # randomize all envs
        indices = torch.arange(self._rwips.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # reward = 1.0 - torch.abs(torch.tanh(4*self.axis_pos)) - 0.01 * torch.abs(self.rxnwheel_vel) * torch.abs(torch.tanh(2*self.axis_pos))
        # reward = 1.0 - torch.abs(torch.tanh(2*self.axis_pos)) - 0.1 * torch.squeeze(torch.abs(torch.mean(self.torque_buffer, dim=0)), dim=1) - 0.005*torch.abs(self.rxnwheel_vel)
        # If we end up outside reset distance, penalize the reward
        angle_term = ((self.axis_pos)/(np.pi/2))**4
        vel_term = (0.01* self.rxnwheel_vel)**2
        # torque_term = 0.1 * torch.squeeze(torch.abs(torch.mean(self.torque_buffer, dim=0)), dim=1)
        # print("ANGLE:", angle_term.cpu().detach().numpy(), "VEL_TERM", vel_term.cpu().detach().numpy(), "TORQUE TERM", torque_term.cpu().detach().numpy())

        if self._log_wandb:
            wandb.log({"Angle Rew Term": angle_term.cpu().detach().numpy(), 
                    "Vel Term": vel_term.cpu().detach().numpy(), 
                    # "Torque Term": torque_term.cpu().detach().numpy() 
                    })
        
        reward = 1.0 - angle_term - vel_term 
        reward = torch.where(torch.abs(self.axis_pos) > 0.5, torch.ones_like(reward) * -10.0, reward)
        self.rew_buf[:] = reward

    def is_done(self) -> None:
        resets = torch.where(torch.abs(self.axis_pos) >= 0.5, 1, 0)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets