import math
import torch
import numpy as np
from typing import Optional

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

from omniisaacgymenvs.tasks.base.rl_task import RLTask

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
        self._max_episode_length = 500

        self._num_observations = 1
        self._num_actions = 1

        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]

    def set_up_scene(self, scene) -> None:
        self.get_rwip()
        super().set_up_scene(scene)
        self._rwips = ArticulationView(
            prim_paths_expr="/World/envs/.*/RWIP", name="rwip_view", reset_xform_properties=False
        )
        scene.add(self._rwips)
        return

    def get_rwip(self):
        rwip = RWIP(prim_path="/RWIP", usd_path="", name="RWIP")
        self._sim_config.apply_articulation_settings(
            "RWIP", get_prim_at_path(rwip.prim_path), self._sim_config.parse_actor_config("RWIP")
        )

    def get_observations(self) -> dict:
        dof_pos = self._rwips.get_joint_positions(clone=False)
        dof_vel = self._rwips.get_joint_velocities(clone=False)

        self.rxnwheel_pos = dof_pos[:, self._rxnwheel_dof_idx]
        self.rxnwheel_vel = dof_vel[:, self._rxnwheel_dof_idx]
        self.axis_pos = dof_pos[:, self._axis_dof_idx]
        self.axis_vel = dof_vel[:, self._axis_dof_idx]

        self.obs_buf[:, 0] = self.rxnwheel_pos
        self.obs_buf[:, 1] = self.rxnwheel_vel
        self.obs_buf[:, 2] = self.axis_pos
        self.obs_buf[:, 3] = self.axis_vel

        observations = {self._rwips.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        pass

    def reset_idx(self, env_ids) -> None:
        pass

    def post_reset(self) -> None:
        # Maybe change dof index names
        self._rxnwheel_dof_idx = self._rwips.get_dof_index("dof_rxnwheel")
        self._axis_dof_idx = self._rwips.get_dof_index("dof_axis")
        # randomize all envs
        indices = torch.arange(self._rwips.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        pass

    def is_done(self) -> None:
        pass