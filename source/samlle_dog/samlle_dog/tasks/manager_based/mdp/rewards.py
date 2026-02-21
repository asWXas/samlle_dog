# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ------------------------ tracking ------------------ #

def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def base_angular_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.linalg.norm((target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-ang_vel_error / std)


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, ramp_at_vel: float = 1.0, ramp_rate: float = 0.5
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm((target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    # fixed 1.0 multiple for tracking below the ramp_at_vel value, then scale by the rate above
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple


# ------------------------ feet ------------------ #

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float,max_time: float
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum(torch.clamp(last_air_time - threshold, min=0.0, max=max_time) * first_contact, dim=1)
    cmd = env.command_manager.get_command(command_name)[:, :2]
    reward *= torch.linalg.norm(cmd, dim=1) > 0.1
    return reward

def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


# feet air time with stepwise reward  离散
def feet_air_time_stepwise(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    min_time: float,    
    max_time: float,  
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    epsilon = 1e-6
    denominator = max(max_time - min_time, epsilon)
    reward_scale = torch.clamp((last_air_time - min_time) / denominator, min=0.0, max=1.0)
    reward = torch.sum(reward_scale * first_contact, dim=1)
    command_norm = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    reward *= (command_norm > 0.1)
    return reward


# 脚在air time窗口内的奖励 稠密
def feet_air_time_window(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    min_time: float = 0.1, 
    max_time: float = 0.5, 
    threshold: float = 1.0
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    is_truly_air = foot_forces_z < threshold
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_window = (current_air_time > min_time) & (current_air_time < max_time) & is_truly_air
    reward = torch.where(in_window, 1.0, 0.0)
    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return torch.sum(reward, dim=1) * (command_norm > 0.1)


# 长时间接触超时惩罚 
def feet_contact_time_long(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg,
    max_time: float = 1.0, 
    threshold: float = 1.0
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces_z = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    current_contact_time = sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    is_overtime = current_contact_time > max_time 
    is_truly_contact = foot_forces_z > threshold
    violation = is_overtime & is_truly_contact
    penalty_score = torch.sum(violation.float(), dim=1)
    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_moving_command = command_norm > 0.1
    return penalty_score * is_moving_command

# 短时间接触超时惩罚
def feet_contact_time_short(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg,
    min_time: float = 0.1, 
    threshold: float = 1.0 # 判定悬空的力阈值
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_contact_time = sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    foot_forces_z = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    is_in_air = foot_forces_z < threshold
    violation_time = (min_time - last_contact_time).clip(min=0.0)
    valid_history = last_contact_time > 0.001
    penalty_per_foot = violation_time * is_in_air.float() * valid_history.float()
    penalty_score = torch.sum(penalty_per_foot, dim=1)
    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :2], dim=1)
    is_moving = command_norm > 0.1
    return penalty_score * is_moving.float()



# 长时间悬空惩罚
# def feet_air_time_long(
#     env: ManagerBasedRLEnv, 
#     command_name: str, 
#     sensor_cfg: SceneEntityCfg,
#     max_time: float = 1.0, 
#     threshold: float = 1.0
# ) -> torch.Tensor:
#     sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#     foot_forces_z = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
#     current_air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
#     is_overtime = current_air_time > max_time 
#     is_truly_air = foot_forces_z < threshold
#     violation = is_overtime & is_truly_air
#     penalty_score = torch.sum(violation.float(), dim=1)
#     command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
#     is_moving_command = command_norm > 0.1
#     return penalty_score * is_moving_command

def feet_air_time_long(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg,
    max_time: float = 1.0, 
    threshold: float = 1.0
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids
    foot_forces_z = sensor.data.net_forces_w[:, body_ids, 2]
    current_air_time = sensor.data.current_air_time[:, body_ids]
    is_truly_air = foot_forces_z < threshold
    overtime = torch.clip(current_air_time - max_time, min=0.0)
    penalty_per_foot = overtime * is_truly_air.float()
    penalty_score = torch.sum(penalty_per_foot, dim=1)
    command = env.command_manager.get_command(command_name)
    command_norm = torch.norm(command[:, :2], dim=1)
    is_moving_command = (command_norm > 0.1).float()
    return penalty_score * is_moving_command

# 单步站立时间奖励
def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


# 打滑
def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

# 打滑惩罚
def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


# 接触
def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return reward

# 踢脚
def feet_stumble(env: ManagerBasedRLEnv, command_name: str,sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return reward

# 抬脚高度  平地有效
def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def scan_foot_clearance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfgs: list[SceneEntityCfg], # 传入4个传感器的配置列表
    asset_cfg: SceneEntityCfg,         # 传入Robot的配置(必须包含body_names)
    target_height: float,
    std: float,
    tanh_mult: float,
    limit: list[float] = [-100.0, 100.0]
) -> torch.Tensor:
    """
    Reward based on 4 separate foot scanners in Isaac Lab.
    
    sensor_cfgs: list[SceneEntityCfg], # 传入4个传感器的配置列表
    asset_cfg: SceneEntityCfg,  pre_order: bool = True
    
    顺序：FL, FR, RL, RR
    
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_heights = []
    for s_cfg in sensor_cfgs:
        sensor: RayCaster = env.scene[s_cfg.name]
        foot_terrain_z = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
        terrain_heights.append(foot_terrain_z)
    terrain_height_batch = torch.stack(terrain_heights, dim=1)
    terrain_height_batch = torch.clamp(terrain_height_batch, min=limit[0], max=limit[1])
    foot_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    clearance = foot_pos_z - terrain_height_batch
    foot_z_target_error = torch.square(clearance - target_height)
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(foot_vel_xy, dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


class FootHeightReward(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # --- 1. 获取资源句柄 ---
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject = env.scene[asset_cfg.name]
        
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        
        # --- 2. 初始化 Buffer ---
        # 预计算脚的数量用于申请内存
        # 注意：这里假设 asset_cfg.body_names 能解析出正确数量的脚
        foot_indices, _ = self.asset.find_bodies(asset_cfg.body_names)
        self.num_feet = len(foot_indices)
        self.dt = env.step_dt

        # 状态变量
        self.min_height = torch.zeros(env.num_envs, self.num_feet, device=env.device)
        self.max_height = torch.zeros(env.num_envs, self.num_feet, device=env.device)
        self.feet_air_time = torch.zeros(env.num_envs, self.num_feet, device=env.device)
        self.last_contacts = torch.zeros(env.num_envs, self.num_feet, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.min_height[env_ids] = 0.0
        self.max_height[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.last_contacts[env_ids] = False

    def __call__(
        self, 
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        target_height: float = 0.05,
        force_thresh: float = 1.0,
        cmd_thresh: float = 0.05,
        command_name: str = "base_velocity"
    ) -> torch.Tensor:
        
        # --- A. 获取数据 ---
        # 1. 脚的高度 (World Frame)
        feet_height = self.asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        
        # 2. 接触力 (Z轴)
        # 警告：确保 sensor_cfg.body_ids 的索引范围在 ContactSensor 数据张量范围内
        contact_force_z = self.contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
        
        # --- B. 状态判定 ---
        is_contact = torch.abs(contact_force_z) > force_thresh
        # 滤波：防止接触瞬间的抖动导致误判
        contact_filt = torch.logical_or(is_contact, self.last_contacts)
        
        # 事件：刚落地 (First Contact) 和 刚起跳 (Just Lifted)
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        just_lifted = (self.feet_air_time == 0.0) & (~contact_filt)

        # --- C. 更新 Min/Max 追踪 ---
        # 刚起跳时，重置 min/max 为当前高度
        self.min_height = torch.where(just_lifted, feet_height, self.min_height)
        self.max_height = torch.where(just_lifted, feet_height, self.max_height)

        # 空中阶段，持续更新极值
        in_air = self.feet_air_time > 0.0
        self.min_height = torch.where(in_air, torch.minimum(self.min_height, feet_height), self.min_height)
        self.max_height = torch.where(in_air, torch.maximum(self.max_height, feet_height), self.max_height)

        # --- D. 计算奖励 (仅在刚落地时触发) ---
        lift_height = self.max_height - self.min_height
        error = lift_height - target_height
        
        # 惩罚项：如果抬腿不够高，reward 为负；如果达标，reward 为 0
        reward_per_foot = torch.clamp(error, max=0.0)
        
        # 汇总：(N, 4) -> (N,)
        rew = torch.sum(reward_per_foot * first_contact, dim=1)

        # --- E. 速度指令 Mask ---
        # 防止机器人原地踏步刷分，只在有水平移动指令时给予奖励/惩罚
        commands = env.command_manager.get_command(command_name)
        moving = torch.norm(commands[:, :2], dim=1) > cmd_thresh
        rew *= moving

        # --- F. 状态迭代 ---
        self.feet_air_time += self.dt
        self.feet_air_time *= (~contact_filt) # 接触则清零
        self.last_contacts = is_contact
        
        # 接触期间清零 min/max，保持状态整洁 (非必须，但推荐)
        self.min_height *= (~contact_filt)
        self.max_height *= (~contact_filt)

        return rew

# --------------------------- Gait --------------------------- #

class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in
    :attr:`synced_feet_pair_names` to bias the policy towards a desired gait, i.e trotting,
    bounding, or pacing. Note that this reward is only for quadrupedal gaits with two pairs
    of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


# ----------------------- joint ------------------ #

# 镜像
def joint_mirror(
    env: ManagerBasedRLEnv, 
    asset_cfg_a: SceneEntityCfg = SceneEntityCfg("robot"), 
    asset_cfg_b: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.25  # 归一化系数，控制奖励的敏感度
) -> torch.Tensor:
    asset_a: Articulation = env.scene[asset_cfg_a.name]
    asset_b: Articulation = env.scene[asset_cfg_b.name]
    curr_joint_pos_a = asset_a.data.joint_pos[:, asset_cfg_a.joint_ids]
    curr_joint_pos_b = asset_b.data.joint_pos[:, asset_cfg_b.joint_ids]
    diff = curr_joint_pos_a - curr_joint_pos_b
    squared_error = torch.sum(torch.square(diff), dim=-1)
    reward = torch.exp(-squared_error / (std**2))
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward

def joint_pos_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(diff), dim=1)

def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel), dim=1)


# ----------------------------- action ------------------ #

def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)



# --------------------------- body ------------------ #


def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)

# 髋关节偏移惩罚
def hip_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚 HAA 关节偏离默认位置。"""
    asset: Articulation = env.scene[asset_cfg.name]
    haa_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    haa_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(haa_pos - haa_default), dim=1)



def base_height_l2_with_limit(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit: list[float] = [-10.0, 10.0],
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    terrain_height = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    terrain_height = torch.clamp(terrain_height, min=limit[0], max=limit[1])
    adjusted_target_height = target_height + terrain_height
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)